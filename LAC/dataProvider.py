"""
DataProvider file
"""
import sys
import random
import utils.data_utils as dtUt
import numpy as np
import logging
from copy import deepcopy

def get_DataProvider(datasetname):
    """
    The function selects the right environment class, according to the used dataset name passed as input and returns
    an instance of this class.
    Or add a function that loads data in this format and then store the full_input

    :param datasetname: for example "Cremi" or "Polygon"
    :type datasetname: str

    :return: An istance of the next two classes DataProvider or CremiDataProvider
    """
    print 'using dataset: ', datasetname
    # This step is just calling the class CremiDataProvider if Cremi is the name of the dataset and returning
    return getattr(sys.modules[__name__], datasetname + "DataProvider")




class DataProvider(object):
    """
    This class loads the dataset and stores it in self.full_input with dim (slices, channels, dimX, dimY).

    Channels can represent raw image and prob. map for instance.
    """
    def __init__(self, input_data_path, label_data_path, slices=None, mirrowBorders=False, pad=None, netFov=None):
        """
        TODO:
            generalize to pick a path for the raw image and one for the membrane prob. map

        :param input_data_path: at the moment this file contains both raw and membrane probability
        :param label_data_path:
        :param slices:
        :type  slices:          ????
        :param mirrowBorders:
        :param pad:             how much to pad the borders
        :param netFov:          field of view of the net, shortcut for pad value --> pad = fov/2
        :type  netFov:          int
        """

        self.log = logging.getLogger(__name__)

        self.input_data_path = input_data_path
        self.label_data_path = label_data_path
        self.slices = slices
        self.log.info("All slices loaded from dataset") if not self.slices else self.log.info("Loaded slices: ", self.slices)

        # Load dataset:
        self.full_input = dtUt.loaddata(str(self.input_data_path), h5_key=None, slices=self.slices)[0]
        self.label = dtUt.loaddata(self.label_data_path, h5_key=None, slices=self.slices)[0]

        # Expand borders (by mirroring image)
        self.mirrorBorders = mirrowBorders
        self.netFov = netFov
        self.pad = pad
        if mirrowBorders:
            if self.netFov:
                self.pad = self.netFov / 2
            assert(self.pad), "Missing pad or netFov value"
            self.full_input = dtUt.mirror_cube(self.full_input, self.pad)
            self.log.warning("Remember to adjust global_patch_size accordingly...")

        # Store some properties of the input:
        self.n_slices = self.full_input.shape[0]
        self.n_channels = self.full_input.shape[1]
        self.dimX = self.full_input.shape[2]
        self.dimY = self.full_input.shape[3]
        self.log.debug(("Loaded dataset: (slices, channels, dimX, dimY):", self.n_slices, self.n_channels, self.dimX, self.dimY))

        # Data augmentation: (not sure if it's still used)
        self.augmentations = [lambda x: x,  # no aug
                               lambda x: x[..., ::-1, :],  # mirror up down
                               lambda x: x[..., :, ::-1],  # mirror left right
                               lambda x: x[..., ::-1, ::-1],  # both
                               lambda x: dtUt.transpose_last(x),  # mirror diagonal
                               lambda x: dtUt.transpose_last(x)[..., ::-1, :],
                              lambda x: dtUt.transpose_last(x)[..., :, ::-1]]
        self.pick_augmentation()


    def pick_augmentation(self):
        """
        Pick one of the random augmentation defined in the init method and store the choice in self.sel_aug_f
        :return:
        """
        self.sel_aug_f = random.choice(self.augmentations)

    def apply_augmentation(self, input):
        """
        Apply the augmentation stored in self.sel_aug_f.

        :param input:
        :return:
        """
        input[:] = self.sel_aug_f(input)

    def get_shape_dataset(self):
        return [self.n_slices, self.n_channels, self.dimX, self.dimY]

    def show_dataset_volumina(self):
        from utils.voluminaView import volumina_n_layer
        image = np.squeeze(self.full_input[:, 0, :, :]).astype(np.float32)
        prob_map = np.squeeze(self.full_input[:, 1, :, :]).astype(np.float32)
        if self.mirrorBorders:
            GTlabels = dtUt.mirror_cube(self.label, self.netFov / 2, mode='constant').astype(np.int8)
        else:
            GTlabels = self.label.astype(np.int8)
        volumina_n_layer([image, prob_map, GTlabels], ["Image", "Prob. map", "Labels"])


class CremiDataProvider(DataProvider):
    pass



class BatchProvider(object):
    """
    This class takes a DataProvider instance, select some parts of the datasets and store a batch.


    Relevant properties:
        - self.boxCoords: coordinates of the selected batch in the image (coord. system could be extended by mirroring)
        - self.GTlabelBoxCoords: coords. in the GT label image (coord. system never mirrored at the borders)
    """
    def __init__(self, dataProvider, batchSize, dimXYpred=0, netFov=None):
        """
        :type dataProvider: DataProvider

        :param dimXYpred: size of the squared box for which we want to get a prediction. By default we take the full dataset.
        :type dimXYpred: int
        """
        self.log = logging.getLogger(__name__)

        self.dataProvider = dataProvider
        self.bs = batchSize
        self.netFov = netFov if netFov else self.dataProvider.netFov
        assert(self.netFov), "Missing network fov"
        self.dimXYpred = dimXYpred
        self.all_XYinput = False
        if self.dimXYpred==0:
            # Take all the dataset xy-size: (e.g. for inference)
            self.all_XYinput = True
            maxDim = self.dataProvider.dimX if self.dataProvider.dimX<=self.dataProvider.dimY else self.dataProvider.dimY
            self.dimXYpred = maxDim - self.netFov + 1

        # Set size box for raw/membrane image:
        self.dimXYimage = self.dimXYpred + self.netFov - 1
        assert(self.dimXYimage <= self.dataProvider.dimX)
        assert(self.dimXYimage <= self.dataProvider.dimY)

        self.batch = None
        self.label = None


    def init_batch(self, preselect_batches=None, quick_eval=True, augment=False):
        """
        Store in the passed variable input the resized version of the input (raw, membrane).

        Options:
            - select specific batches (i.e. certain z-slices)
            - restrict the xy dimension of the prediction

        :param input:
        :param preselect_batches: list of which slices select
        :return:
        """
        self.batch = np.empty(self.get_input_batch_shape())
        self.quick_eval = quick_eval
        self.augment = augment
        self.preselect_batches = preselect_batches

        ''' Which z-slices should we select? '''
        if preselect_batches is not None:
            self.log.info(("Using preselected batches: ",preselect_batches))
            assert (self.bs == len(preselect_batches))
            ind_b = preselect_batches
        elif self.quick_eval:
            n_z = self.dataProvider.n_slices
            ind_b = np.linspace(0, n_z, self.bs, dtype=np.int, endpoint=False)
            self.log.info(("Using fixed batches with z-slices equally distributed: ", ind_b) )
        else:
            ind_b = np.random.permutation(range(self.dataProvider.n_slices))[:self.bs]
            self.log.warning(("Pay attention to this: is it always correct?", ind_b))
            if augment:
                self.dataProvider.pick_augmentation()


        ''' Should we restrict the xy dimension of the input? '''
        if not self.all_XYinput:
            # Decide where to center the box:
            if self.quick_eval:
                # Pick it in the top-left corner:
                self.log.info('Using fixed indices at the corner')
                ind_x = np.empty(self.bs, dtype=int)
                ind_x.fill(int(0))
                ind_y = np.empty(self.bs, dtype=int)
                ind_y.fill(int(0))
            else:
                # Choose randomly:
                ind_x = np.random.randint(0,
                                          self.dataProvider.dimX - self.dimXYimage + 1,
                                          size=self.bs)
                ind_y = np.random.randint(0,
                                          self.dataProvider.dimY - self.dimXYimage + 1,
                                          size=self.bs)
            for b in range(self.bs):
                self.batch[b, :, :, :] = self.dataProvider.full_input[ind_b[b], :,
                                         ind_x[b]:ind_x[b] + self.dimXYimage,
                                         ind_y[b]:ind_y[b] + self.dimXYimage]
            if self.augment:
                self.dataProvider.apply_augmentation(self.batch)

            self.boxCoords = (ind_b, ind_x, ind_y)
        else:
            self.batch[range(self.bs)] = self.dataProvider.full_input[ind_b]
            if self.augment:
                self.dataProvider.apply_augmentation(self.batch)
            self.boxCoords = (ind_b, None, None)

        # Compute GTlabel_batch:
        self.init_GTlabel_batch()
        return list(self.boxCoords)


    def init_GTlabel_batch(self):
        self.label = np.empty(self.get_GTlabelBatch_shape())

        if not self.all_XYinput:
            self.GTlabelBoxCoords = deepcopy(self.boxCoords)
            ind_b, ind_x, ind_y = self.GTlabelBoxCoords
            # Adjust prediction box: (different coordinate system)
            if not self.dataProvider.mirrorBorders:
                ind_x += self.netFov / 2
                ind_y += self.netFov / 2
            self.log.debug(("Return GT labels at coord.:", ind_b, ind_x, ind_y))
            self.log.debug(("Raw image position:", self.boxCoords))
            for b in range(self.bs):
                self.label[b, :, :] = self.dataProvider.label[ind_b[b],
                                      ind_x[b]:ind_x[b] + self.dimXYpred,
                                      ind_y[b]:ind_y[b] + self.dimXYpred]
            if self.augment:
                self.dataProvider.apply_augmentation(self.label)
        else:
            for b in range(self.bs):
                # Take full image: (or pad if raw image was not mirrored)
                if self.dataProvider.mirrorBorders:
                    self.label[b] = self.dataProvider.label[b, :, :]
                else:
                    self.label[b] = self.dataProvider.label[b, self.dataProvider.pad:-self.dataProvider.pad,
                                    self.dataProvider.pad:-self.dataProvider.pad]
            if self.augment:
                self.dataProvider.apply_augmentation(self.label)

    def get_input_batch_shape(self):
        """
        Helper function: THIS DIMENSION INCLUDES BOTH THE RAW IMAGE AND THE PROB. MAP
        :return: dim(batch_size, channels, dimX, dimY)
        """
        data_shape = list(self.dataProvider.full_input.shape)
        data_shape[0] = self.bs
        if not self.all_XYinput:
            data_shape[2] = self.dimXYimage
            data_shape[3] = self.dimXYimage
        return data_shape


    def get_image_shape(self):
        """
        Helper function: just the dimension of the raw image (without membranes or other channels)
        :return: dim(batch_size, dimX, dimY)
        """

        data_shape = list(self.get_input_batch_shape())
        del data_shape[1]
        return data_shape

    def get_GTlabelBatch_shape(self):
        """
        Helper function
        :return:
        """
        data_shape = list(self.dataProvider.label.shape)
        data_shape[0] = self.bs

        if not self.all_XYinput:
            data_shape[1] = self.dimXYpred
            data_shape[2] = self.dimXYpred

        return data_shape


    def crop_batch_XY(self, center, b, out=None):
        """
        Crop the batch XY-input according to the netFov given a center coordinates.
        Node based, so the netFov should be odd.

        :param center: (x_coord, y_coord)
        :param b: batch number
        :param out:
        :return:
        """
        assert(self.netFov%2!=0, "Even netFov, center pixel not defined")
        pad = self.netFov / 2
        if out is None:
            return self.batch[b, :,
                        center[0]-pad : center[0]+pad+1,
                        center[1]-pad : center[1]+pad+1]
        else:
            out[:] = self.batch[b, :,
                        center[0]-pad : center[0]+pad+1,
                        center[1]-pad : center[1]+pad+1]


    def show_batch_volumina(self):
        from utils.voluminaView import volumina_n_layer
        image = self.batch[:,0,:,:].astype(np.float32)
        prob_map = self.batch[:, 1, :, :].astype(np.float32)
        GTlabels = dtUt.mirror_cube(self.label,self.netFov/2,mode='constant').astype(np.int8)
        volumina_n_layer([image, prob_map, GTlabels], ["Image", "Prob. map", "Labels"])


    def get_seed_coords_from_file(self, global_seeds):
        '''
        DEPRECATED
        No longer used....????

        :param global_seeds:
        :return:
        '''
        self.log.error("get_seed_coords_from_file() not checked")
        # clear global_seeds but keep empty list reference
        seeds = dtUt.loaddata(str(self.options.seed_file_path))
        del global_seeds[:]
        for b in range(self.bs):
            self.global_seeds.append(seeds[b] + self.pad)


class LAC_BatchProvider(BatchProvider):
    """
    Adds function for computing affinity graph (for the moment only in 2D)
    TODO:
        - affinity map
        - crop image in one center (implement in the upper class)
    """

    def init_batch(self, preselect_batches=None, quick_eval=True, augment=False):
        """
        Compute affinity graph and replace the prob. map in the batch data
        """

        # Init batch and GT_labels:
        super(LAC_BatchProvider, self).init_batch(preselect_batches, quick_eval, augment)

        # Compute affinities:
        nDims_affinities = 2 # 2D for the moment
        shape_affinities = list(self.batch.shape)
        shape_affinities[1] = nDims_affinities
        self.batch_prob_map = self.batch[:, 1, :, :]

        self.log.info("Temporary redundant storage of affinities and prob_map")
        self.affinities = np.empty(shape_affinities)
        self.log.warning("Check the def. of the affinities")
        self.log.warning("Last affinity is wrong...")
        for axis in range(nDims_affinities):
            self.affinities[:,axis,...] = self.batch_prob_map - np.roll(self.batch_prob_map,-1,axis=axis+1)

        self.batch = np.concatenate((self.batch, self.affinities),axis=1)
        self.batch = np.delete(self.batch,1,axis=1) # Delete prob. map


    def crop_batch_XY(self, centerPred, b, out=None):
        """
        TODO: UPDATE WITH LIST OF CENTERS (one for each batch)

        Crop the batch XY-input according to the netFov given a center coordinates of a selected edge.

        REMARK: centerPred are the coordinates of the selected edge in the prediction coord. system
                (not in the padded raw-image/edges)


        The coordinates of the center pixel and the selected edge are equivalent:

            - if an edge along x is selected: (selected edge: tilted one)

                x - x - x - x - x -
                |   |   |   |   |
                x - x - x - x - x -
                |   |   |   |   |
                x - x - O - x - x -
                |   |   \   |   |
                x - x - x - x - x -
                |   |   |   |   |
                x - x - x - x - x -
                |   |   |   |   |


            - if an edge along y is selected: (selected edge: highlighted one)

                x - x - x - x - x -
                |   |   |   |   |
                x - x - x - x - x -
                |   |   |   |   |
                x - x - O = x - x -
                |   |   |   |   |
                x - x - x - x - x -
                |   |   |   |   |
                x - x - x - x - x -
                |   |   |   |   |

        In both the previous cases center = (shift_x+2, shift_y+2)

        :param center: xy-coordinates of the central pixel/selected edge
        :param b: batch number
        :param out:
        """
        assert(self.netFov%2!=0, "Even netFov, center pixel not defined")
        pad = self.netFov / 2
        center = np.array(list(centerPred)) + self.netFov / 2

        if out is None:
            return self.batch[b, :,
                        center[0]-pad : center[0]+pad+1,
                        center[1]-pad : center[1]+pad+1]
        else:
            out[:] = self.batch[b, :,
                        center[0]-pad : center[0]+pad+1,
                        center[1]-pad : center[1]+pad+1]


    def show_batch_volumina(self, prob_map=False):

        image = self.batch[:, 0, :, :].astype(np.float32)
        aff_x = self.batch[:,1,...].astype(np.float32)
        aff_y = self.batch[:,2,...].astype(np.float32)
        GTlabels = dtUt.mirror_cube(self.label, self.netFov / 2, mode='constant').astype(np.int8)

        from utils.voluminaView import volumina_n_layer
        if prob_map:
            prob_map = self.batch_prob_map.astype(np.float32)
            volumina_n_layer([image, prob_map, aff_x, aff_y, GTlabels], ["Image", "Prob. map", "aff_x", "aff_y", "Labels"])
        else:
            volumina_n_layer([image, aff_x, aff_y, GTlabels], ["Image", "aff_x", "aff_y", "Labels"])


def show_cropped_batch_volumina(cropped_batch):
    image = cropped_batch[:, 0, ...].astype(np.float32)
    aff_x = cropped_batch[:, 1, ...].astype(np.float32)
    aff_y = cropped_batch[:, 2, ...].astype(np.float32)
    from utils.voluminaView import volumina_n_layer
    volumina_n_layer([image, aff_x, aff_y], ["Image", "aff_x", "aff_y"])

# class croppedBatch(object):
#     def __init__(self):



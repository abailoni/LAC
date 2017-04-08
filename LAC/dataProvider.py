"""
DataProvider file
"""
import sys
import random
import utils.data_utils as dtUt
import numpy as np
import logging
from copy import deepcopy
import warnings


def get_DataProvider(typeDataProvider):
    """
    OUTDATED DOCS


    The function selects the right environment class, according to the used dataset name passed as input and returns
    an instance of this class.
    Or add a function that loads data in this format and then store the raw_image

    :param typeDataProvider: for example "Cremi" or "Polygon"
    :type typeDataProvider: str

    :return: An istance of the next two classes DataProvider or CremiDataProvider
    """
    print 'using dataset: ', typeDataProvider
    # This step is just calling the class CremiDataProvider if Cremi is the name of the dataset and returning
    return getattr(sys.modules[__name__], typeDataProvider + "DataProvider")





class DataProvider(object):
    """
    Generic DataProvider loading raw image, prob. map and GT batchLabels.

    Probability map is optional.
    """
    def __init__(self,
                 input_data_path,
                 probMap_data_path=None,
                 slices=None,
                 mirrowBorders=False,
                 pad=None,
                 netFov=None,
                 h5path_rawImage='volumes/raw',
                 h5path_labels='volumes/labels/neuron_ids',
                 probMap_data_h5path='data'):
        """

        :param input_data_path:     it contains both the image and the batchLabels (CREMI style)
        :param membProb_data_path:  path for the computed membrane probability map
        :param slices:
        :param mirrowBorders:
        :param pad:             how much to pad the borders
        :param netFov:          field of view of the net, shortcut for pad value --> pad = fov/2
        :type  netFov:          int
        """

        self.log = logging.getLogger(__name__)

        self.slices = slices
        self.log.info("All slices loaded from dataset") if not self.slices else self.log.info("Loaded slices: ", self.slices)

        # Load dataset:
        self.raw_image = dtUt.loaddata(str(input_data_path), h5_key=h5path_rawImage, slices=self.slices)[0]
        self.label = dtUt.loaddata(input_data_path, h5_key=h5path_labels, slices=self.slices)[0]
        assert (self.raw_image.shape==self.label.shape)
        self.probMap = None
        if probMap_data_path:
            self.probMap = dtUt.loaddata(probMap_data_path, h5_key=probMap_data_h5path, slices=self.slices)[0]
            assert(self.raw_image.shape==self.probMap.shape)
            self.log.info("Probability map loaded correctly")

        # Expand borders (by mirroring image)
        self.mirrorBorders = mirrowBorders
        self.netFov = netFov
        self.pad = pad
        if mirrowBorders:
            if self.netFov:
                self.pad = self.netFov / 2
            assert(self.pad), "Missing pad or netFov value"
            self.raw_image = dtUt.mirror_cube(self.raw_image, self.pad)
            if self.probMap:
                self.probMap = dtUt.mirror_cube(self.probMap, self.pad)
            self.log.warning("Remember to adjust global_patch_size accordingly...")


        self.num_slices = self.raw_image.shape[0]
        self.sizeX_raw = self.raw_image.shape[1]
        self.sizeY_raw = self.raw_image.shape[1]
        self.log.debug(("Loaded dataset: raw data (slices, dimX, dimY):", self.raw_image.shape))
        self.log.debug(("Loaded dataset: batchLabels (slices, dimX, dimY):", self.label.shape))
        self.log.debug(("Loaded dataset: mirroring borders ", self.mirrorBorders))

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
        return self.raw_image.shape

    def get_shape_labels(self):
        return self.labels.shape


    def show_dataset_volumina(self):
        from utils.voluminaView import volumina_n_layer
        image = self.raw_image.astype(np.float32)
        if self.mirrorBorders:
            GTlabels = dtUt.mirror_cube(self.label, self.netFov / 2, mode='constant').astype(np.int8)
        else:
            GTlabels = self.label.astype(np.int8)

        if self.probMap:
            prob_map = self.probMap.astype(np.float32)
            volumina_n_layer([image, prob_map, GTlabels], ["Image", "Prob. map", "Labels"])
        else:
            volumina_n_layer([image, GTlabels], ["Image", "Labels"])


    def outdated_computation_affinities(self):
        assert(self.probMap, "Probability map is required to compute affinities")

        # Compute affinities:
        nDims_affinities = len(self.raw_image.shape)
        shape_affinities = [nDims_affinities] + list(self.raw_image.shape)

        self.affinities = np.empty(shape_affinities)
        self.log.warning("WRONG DEF OF AFFINITIES!!!")
        self.log.warning("Last affinity in the block is wrong...")
        for axis in range(nDims_affinities):
            self.affinities[axis,...] = self.probMap - np.roll(self.probMap,-1,axis=axis)




class AffinityDataProvider(DataProvider):
    """
    Sub class of DataProvider with the added possibility to store precomputed affinities maps.
    """
    def __init__(self, input_data_path, **kwargs):
        """
        It accepts the following extra parameters:

            - affinity_data_path (mandatory)
            - affinity_data_h5path (optional). Default is 'data'

        :param args:
        :param kwargs:
        """
        if 'affinity_data_path' not in kwargs:
            raise TypeError('Missing required field affinity_data_path')
        affinity_data_path = kwargs.pop('affinity_data_path')
        affinity_data_h5path = kwargs.pop('affinity_data_h5path', 'data')
        super(AffinityDataProvider, self).__init__(input_data_path, **kwargs)

        # Load affinity data:
        self.affinities = dtUt.loaddata(str(affinity_data_path), h5_key=affinity_data_h5path, slices=self.slices)[0]
        if self.mirrorBorders:
            self.affinities = dtUt.mirror_cube(self.affinities, self.pad)

    def show_dataset_volumina(self):
        from utils.voluminaView import volumina_n_layer
        image = self.raw_image.astype(np.float32)
        if self.mirrorBorders:
            GTlabels = dtUt.mirror_cube(self.label, self.netFov / 2, mode='constant').astype(np.int8)
        else:
            GTlabels = self.label.astype(np.int8)

        affX, affY, affZ = self.affinities[0], self.affinities[1], self.affinities[2]

        maps, maps_labels = [], []
        if self.probMap:
            maps += [self.probMap.astype(np.float32)]
            maps_labels += ['Prob. map']

        volumina_n_layer([image]+maps+[affX, affY, affZ, GTlabels], ["Image"]+maps_labels+["Aff. x","Aff. y","Aff. z","Labels"])




class StaticBatchProvider2D:
    """
    This class takes a DataProvider instance, select some parts of the datasets and store a batch.
    This batch will become the static/initial part of the envStatus (raw image/affinities/GTlabels).

    REMARK:
      - for the moment it works for 2D samples (with possible z-context), but not for 3D cubes.

    Properties:
        - self.staticBatch useful for initializing graph weights
        - self.staticBatch_padded (should not be necessary)
        - self.batchLabels: useful for storing GT in the graph (not padded)
        - use get_cropped_staticBatch() to get the padded-cropped inputs for the net. Labels at this stage
            should not be necessary
    """
    def __init__(self, dataProvider, batchSize, sizeXYpred=(0,0), netFov=None, zContext=None):
        """
        :type dataProvider: AffinityDataProvider

        :param sizeXYpred: size of the box for which we want to get a prediction. By default we take the full dataset.
        :type sizeXYpred: int or list
        """
        # TODO: add z-context option
        self.log = logging.getLogger(__name__)

        assert (isinstance(dataProvider,AffinityDataProvider))
        self.dataProvider = dataProvider
        self.bs = batchSize
        if zContext:
            raise NotImplemented("Only implementation available: one slice selected")
        self.netFov = netFov if netFov else self.dataProvider.netFov

        assert(self.netFov), "Missing network fov"
        assert (self.netFov % 2 != 0, "Even netFov, center pixel not defined")

        self.all_XYinput = False
        if sizeXYpred==(0,0):
            # Take all the dataset xy-size: (e.g. for inference)
            if not self.dataProvider.mirrorBorders:
                warn_message = "Borders were not mirrored. The batch will include the full dataset, but prediction will \
                                exclude some border pixels. Activate mirrorBorders for having a full-size prediction"
                warnings.warn(warn_message)
                self.log.warning(warn_message)
            self.all_XYinput = True
            self.sizeXYraw = np.array([self.dataProvider.sizeX_raw, self.dataProvider.sizeY_raw])
            self.sizeXYpred = self.sizeXYraw - (self.netFov - 1)
        else:
            if isinstance(sizeXYpred, int):
                sizeXYpred = [sizeXYpred, sizeXYpred]
            self.sizeXYpred = np.array(sizeXYpred)
            self.sizeXYraw = self.sizeXYpred + (self.netFov - 1)
            assert(self.sizeXYraw[0] <= self.dataProvider.sizeX_raw)
            assert(self.sizeXYraw[1] <= self.dataProvider.sizeY_raw)



    def init_staticBatch(self,
                   preselect_slices=None,
                   typeInputMap='affinities',
                   quick_eval=True,
                   augment=False):
        """
        Options:
            - select specific batches (i.e. certain z-slices)
            - restrict the xy dimension of the prediction

        :param preselect_slices: list of which slices select
        :return:
        """

        if typeInputMap!='affinities':
            raise NotImplemented('Other methods not implemented')


        self.staticBatch_padded = np.empty(self.get_staticBatch_padded_shape())
        self.quick_eval = quick_eval
        self.augment = augment
        self.preselect_slices = preselect_slices

        ''' Which z-slices should we select? '''
        self.log.debug("Selecting batch-Z-positions in the dataset:")
        if preselect_slices is not None:
            self.log.info(("Using preselected batches: ", preselect_slices))
            assert (self.bs == len(preselect_slices))
            ind_b = preselect_slices
        elif self.quick_eval:
            n_z = self.dataProvider.num_slices
            ind_b = np.linspace(0, n_z, self.bs, dtype=np.int, endpoint=False)
            self.log.info(("Using fixed batches with z-slices equally distributed: ", ind_b) )
        else:
            ind_b = np.random.permutation(range(self.dataProvider.num_slices))[:self.bs]
            self.log.warning(("Random selected slices: is it always correct...?", ind_b))
            if augment:
                self.dataProvider.pick_augmentation()


        ''' Should we restrict the xy dimension of the input? '''
        self.log.debug("Selecting batch-XY-positions in the dataset")
        if not self.all_XYinput:
            # Decide where to center the box:
            if self.quick_eval:
                # Pick it in the top-left corner:
                self.log.info('Sampling at the top-left corner of the dataset')
                ind_x = np.zeros(self.bs, dtype=int)
                ind_y = np.zeros(self.bs, dtype=int)
            else:
                # Choose randomly:
                self.log.info("Sampling randomly in the XY dimension")
                ind_x = np.random.randint(0,
                                          self.dataProvider.sizeX_raw - self.sizeXYraw[0] + 1,
                                          size=self.bs)
                ind_y = np.random.randint(0,
                                          self.dataProvider.sizeY_raw - self.sizeXYraw[1] + 1,
                                          size=self.bs)
            for b in range(self.bs):
                # Raw image:
                self.staticBatch_padded[b, 0, :, :] = self.dataProvider.raw_image[ind_b[b],
                                                      ind_x[b]:ind_x[b]+self.sizeXYraw[0],
                                                      ind_y[b]:ind_y[b]+self.sizeXYraw[1]]
                self.staticBatch_padded[b, 1:, :, :] = self.dataProvider.affinities[:, ind_b[b],
                                                       ind_x[b]:ind_x[b] + self.sizeXYraw[0],
                                                       ind_y[b]:ind_y[b] + self.sizeXYraw[1]]
        else:
            self.log.info("All XY size taken")
            ind_x = np.zeros(self.bs, dtype=int)
            ind_y = np.zeros(self.bs, dtype=int)
            self.staticBatch_padded[range(self.bs), 0, ...] = self.dataProvider.raw_image[ind_b]
            self.staticBatch_padded[range(self.bs), 1:, ...] = self.dataProvider.affinities[:, ind_b, ...]

        self.boxCoords = (ind_b, ind_x, ind_y)
        if self.augment:
            self.dataProvider.apply_augmentation(self.staticBatch_padded)

        self.pad = self.netFov/2
        self.staticBatch = self.staticBatch_padded[...,
                           self.pad:-self.pad,
                           self.pad:-self.pad]

        # Compute GTlabel_batch:
        self._init_batchLabels()
        return list(self.boxCoords)


    def _init_batchLabels(self):
        self.batchLabels = np.empty(self.get_predictionLabels_shape(), dtype=np.int32)

        if not self.all_XYinput:
            self.GTlabelBoxCoords = deepcopy(self.boxCoords)
            ind_b, ind_x, ind_y = self.GTlabelBoxCoords
            # Adjust prediction box: (different coordinate system)
            if not self.dataProvider.mirrorBorders:
                ind_x += self.netFov / 2
                ind_y += self.netFov / 2
            self.log.debug(("Return GT batchLabels at coord.:", ind_b, ind_x, ind_y))
            self.log.debug(("Raw image position:", self.boxCoords))
            for b in range(self.bs):
                self.batchLabels[b, :, :] = self.dataProvider.label[ind_b[b],
                                            ind_x[b]:ind_x[b] + self.sizeXYpred[0],
                                            ind_y[b]:ind_y[b] + self.sizeXYpred[1]]
        else:
            for b in range(self.bs):
                # Take full image: (or pad if raw image was not mirrored)
                if self.dataProvider.mirrorBorders:
                    self.batchLabels[b] = self.dataProvider.label[b, :, :]
                else:
                    self.batchLabels[b] = self.dataProvider.label[b,
                                          self.dataProvider.pad:-self.dataProvider.pad,
                                          self.dataProvider.pad:-self.dataProvider.pad]
        if self.augment:
            self.dataProvider.apply_augmentation(self.batchLabels)


        # Do I really need these...?
        self.batchLabels_padded = dtUt.mirror_cube(self.batchLabels, self.netFov / 2, mode='constant', constant_values=-9999)

    def get_staticBatch_padded_shape(self):
        """
        :return: dim(batch_size, channels, padded_sizeX, padded_sizeY)

        Here channels usually is raw+affinities=4, but there could be z-context.

        This batch should actually never be useful externally. The not padded version of the batch will be used
        at the beginning in the initializiation of the graph.
        """
        return [self.bs]+[1+self.dataProvider.affinities.shape[0]]+list(self.sizeXYraw)


    def get_predictionLabels_shape(self):
        """
        :return: dim(batch_size, sizeX, sizeY)

        (Not padded XY dimensions)
        """
        return [self.bs]+list(self.sizeXYpred)

    def get_cropped_staticBatch(self, selected_edgesCoorXY, out=None, b=None):
        """
        Crop the batch XY-input according to the netFov given a center coordinates of a selected edge.

        REMARK: centerPred are the coordinates of the selected edge in the prediction coord. system
                (not in the padded raw/affinities)


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

        Two modes:
            - selected edges for all batches --> b=None, selected_edgesCoorXY.shape = (bs, 2)
            - selected edges for a specific batches b --> selected_edgesCoorXY.shape = (num_edges, 2)

        :param selected_edgesCoorXY: xy-coordinates of the central pixel/selected edge
        :type selected_edgesCoorXY: array of shape (2, batch_size)
        :param out:

        :param b: array with associoted batches indices
        """
        assert(selected_edgesCoorXY.shape[1]==2)

        if b is None:
            assert(selected_edgesCoorXY.shape[0]==self.bs)
        else:
            assert (selected_edgesCoorXY.shape[0] == b.shape[0])

        pad = self.netFov / 2
        edgesPadCoor = selected_edgesCoorXY + pad


        if out is None:
            if b is None:
                out = np.empty(self.get_cropped_staticBatch_shape())
            else:
                shape = self.get_cropped_staticBatch_shape()
                shape[0] = b.shape[0]
                out = np.empty(shape)

        if b is None:
            b = range(self.bs)

        for batch in b:
            out[:] = self.staticBatch_padded[batch, :,
                        edgesPadCoor[batch,0]-pad : edgesPadCoor[batch,0]+pad+1,
                        edgesPadCoor[batch,1]-pad : edgesPadCoor[batch,1]+pad+1]
        return out

    def get_cropped_staticBatch_shape(self):
        shape = self.get_staticBatch_padded_shape()
        shape[2] = self.netFov
        shape[3] = self.netFov
        return shape

    # def show_batch_volumina(self):
    #     from utils.voluminaView import volumina_n_layer
    #     image = self.staticBatch_padded[:,0,:,:].astype(np.float32)
    #     prob_map = self.batch[:, 1, :, :].astype(np.float32)
    #     GTlabels = dtUt.mirror_cube(self.batchLabels, self.netFov / 2, mode='constant').astype(np.int8)
    #     volumina_n_layer([image, prob_map, GTlabels], ["Image", "Prob. map", "Labels"])

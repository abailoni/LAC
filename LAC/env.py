"""
Environment file
"""
import sys
import random
import utils.data_utils as dtUt
import numpy as np

def get_DataProvider(datasetname):
    """
    The function selects the right environment class, according to the used dataset name passed as input and returns
    an instance of this class.

    :param datasetname: for example "Cremi" or "Polygon"
    :type datasetname: str

    :return: An istance of the next two classes DataProvider or CremiDataProvider
    """
    print('using dataset: ', datasetname)
    # This step is just calling the class CremiDataProvider if Cremi is the name of the dataset and returning
    return getattr(sys.modules[__name__], datasetname + "DataProvider")


class DataProvider(object):
    """
    This class loads the dataset and stores it in self.full_input

    Options:
        - loads only some of the z-slices


    """
    def __init__(self, options, netOptions):
        # TODO:
        #   - delete all the part related to the selection of a portion of the dataset
        #   - I don't like this thjing that the actual global_width depends on padding or not...
        #   - put everything in another class, called something like envStatus
        #   - need to decide if I should move this in another fun or not
        #   - for sure it would be nice to have all these un-necessary options away from the init method and input only
        #       when I decide to select a specific part (or are they global...?)
        #       Create at least a subclass that does this, at least there is one that is really general (just load the
        #       dataset)

        self.options = options
        self.bs = options.batch_size
        self.set_slices(options)
        self.netFov = netOptions['netFov']

        # Padding that I can add at the borders of the netFov:
        self.pad = netOptions['netFov'] / 2

        # WHAT IS THIS....??
        # Probably something like the selected area in the dataset...?
        self.options.global_input_len = self.options.global_edge_len  # modified in load data

        # Load the full input:
        # the final shape of self.full_input is [z_slices, channels, x_dim, y_dim]
        # Two possible channels are the following:
        #     - raw image
        #     - probability map
        self.full_input = None
        self.load_data(options)

        # Some properties of the input:
        self.n_slices = self.full_input.shape[0]
        self.dimX = self.full_input.shape[2]
        self.dimY = self.full_input.shape[3]
        print ("patch_len, global_edge_len, self.rlxy", self.netFov, options.global_edge_len, self.dimX, self.dimY)

        assert (options.global_input_len <= self.dimX)

        self.augmentations = [lambda x: x,  # no aug
                               lambda x: x[..., ::-1, :],  # mirror up down
                               lambda x: x[..., :, ::-1],  # mirror left right
                               lambda x: x[..., ::-1, ::-1],  # both
                               lambda x: dtUt.transpose_last(x),  # mirror diagonal
                               lambda x: dtUt.transpose_last(x)[..., ::-1, :],
                              lambda x: dtUt.transpose_last(x)[..., :, ::-1]]
        self.pick_augmentation()

    def load_data(self, options):
        """
        The method loads the full image of the datasets.

        The shape of self.full_input is [z_slices, channels, x_dim, y_dim].

        :param options:
        :return: None
        """
        self.full_input = dtUt.loaddata(str(self.options.input_data_path), h5_key=None, slices=self.slices)[0]

        # Decide if to mirror the image at t
        if options.padding_b:
            self.full_input = dtUt.mirror_cube(self.full_input, self.pad)
            self.options.global_input_len += self.netFov - 1

        self.label = dtUt.loaddata(self.options.label_path, h5_key=None, slices=self.slices)[0]


    def set_slices(self, options):
        """
        Create an attribute slices from the options with a list of which slices select from the original
        h5 dataset.

        :param options:
        :return:
        """
        if "slices" in options:
            self.slices = options.slices
        else:
            self.slices = None
            print("no slices selected")

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

    def prepare_input_batch(self, input, preselect_batches=None):
        """
        Store in the passed variable input the resized version of the input (raw, membrane).

        Options:
            - select specific batches (i.e. certain z-slices)
            - restrict the xy dimension of the prediction

        :param input:
        :param preselect_batches:
        :return:
        """

        ''' Which z-slices should we select? '''
        if preselect_batches is not None:
            print("Using fixed batches bs", self.bs, 'preselected slices', preselect_batches)
            assert (self.bs == len(preselect_batches))
            ind_b = preselect_batches
        elif self.options.quick_eval:
            print("Using fixed batches equally distributed")
            n_z = self.full_input.shape[0]
            ind_b = np.linspace(0, n_z, self.bs, dtype=np.int, endpoint=False)
        else:
            ind_b = np.random.permutation(range(self.n_slices))[:self.bs]
            if self.options.augment_ft:
                self.pick_augmentation()


        ''' Should we restrict the xy dimension of the input? '''
        if self.options.global_edge_len > 0:
            ''' Here we restrict the input to a squared window of dim '''
            # Decide where to center this window:
            if self.options.quick_eval:
                # Pick it in the top-left corner:
                print('using fixed indices')
                ind_x = np.empty(self.bs, dtype=int)
                ind_x.fill(int(0))
                ind_y = np.empty(self.bs, dtype=int)
                ind_y.fill(int(0))
            else:
                # Choose randomly:
                ind_x = np.random.randint(0,
                                          self.dimX - self.options.global_input_len + 1,
                                          size=self.bs)
                ind_y = np.random.randint(0,
                                          self.dimY - self.options.global_input_len + 1,
                                          size=self.bs)
            for b in range(self.bs):
                input[b, :, :, :] = self.full_input[ind_b[b], :,
                                    ind_x[b]:ind_x[b] + self.options.global_input_len,
                                    ind_y[b]:ind_y[b] + self.options.global_input_len]
            if self.options.augment_ft:
                self.apply_augmentation(input)

            return [ind_b, ind_x, ind_y]
        else:
            input[range(self.bs)] = self.full_input[ind_b]
            if self.options.augment_ft:
                self.apply_augmentation(input)
            return [ind_b, None, None]

    def prepare_label_batch(self, label, height, rois):
        if self.options.global_edge_len > 0:
            ind_b, ind_x, ind_y = rois
            if not self.options.padding_b:
                ind_x += self.netFov / 2
                ind_y += self.netFov / 2
            for b in range(self.bs):
                label_inp_len = self.options.global_input_len - self.netFov + 1
                height[b, :, :] = self.height_gt[ind_b[b],
                                  ind_x[b]:ind_x[b] + label_inp_len,
                                  ind_y[b]:ind_y[b] + label_inp_len]
                label[b, :, :] = self.label[ind_b[b],
                                 ind_x[b]:ind_x[b] + label_inp_len,
                                 ind_y[b]:ind_y[b] + label_inp_len]
            if self.options.augment_ft:
                self.apply_augmentation(height)
                self.apply_augmentation(label)
        else:
            for b in range(self.bs):
                if self.options.padding_b:
                    label[b] = self.label[b, :, :]
                    height[b] = self.height_gt[b, :, :]
                else:
                    label[b] = self.label[b, self.pad:-self.pad,
                               self.pad:-self.pad]
                    height[b] = self.height_gt[b, self.pad:-self.pad,
                                self.pad:-self.pad]
            if self.options.augment_ft:
                self.apply_augmentation(height)
                self.apply_augmentation(label)

    def get_seed_coords_from_file(self, global_seeds):
        '''
        No longer used....????

        :param global_seeds:
        :return:
        '''
        # clear global_seeds but keep empty list reference
        seeds = dtUt.loaddata(str(self.options.seed_file_path))
        del global_seeds[:]
        for b in range(self.bs):
            self.global_seeds.append(seeds[b] + self.pad)




    def get_batch_shape(self):
        """
        Helper function
        :return: dim(batch_size, channels, dimX, dimY)
        """
        data_shape = list(self.full_input.shape)
        data_shape[0] = self.bs
        if self.options.global_edge_len > 0:
            data_shape[2] = self.options.global_input_len
            data_shape[3] = self.options.global_input_len

        return data_shape

    def get_image_shape(self):
        """
        Helper function: just the dimension of the raw image (without membranes or other channels)
        :return: dim(batch_size, dimX, dimY)
        """

        data_shape = list(self.get_batch_shape())
        del data_shape[1]
        return data_shape

    def get_label_shape(self):
        """
        Helper function
        :return:
        """
        data_shape = list(self.label.shape)
        data_shape[0] = self.bs

        if self.options.global_edge_len > 0:
            data_shape[1] = self.options.global_edge_len
            data_shape[2] = self.options.global_edge_len

        if not self.options.padding_b:
            data_shape[1] -= self.netFov - 1
            data_shape[2] -= self.netFov - 1

        return data_shape


class CremiDataProvider(DataProvider):
    pass

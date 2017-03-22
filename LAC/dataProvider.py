"""
DataProvider file
"""
import sys
import random
import utils.data_utils as dtUt
import numpy as np
import logging
from copy import deepcopy

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
    Generic DataProvider loading raw image, prob. map and GT labels.

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

        :param input_data_path:     it contains both the image and the labels (CREMI style)
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
            log.info("Probability map loaded correctly")

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

        # Store some properties of the input:
        self.n_slices = self.raw_image.shape[0]
        self.dimX = self.raw_image.shape[1]
        self.dimY = self.raw_image.shape[2]
        self.log.debug(("Loaded dataset: (slices, dimX, dimY):", self.n_slices, self.dimX, self.dimY))

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
        return [self.n_slices, self.dimX, self.dimY]

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



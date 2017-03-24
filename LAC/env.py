"""
Here we construct the specific environment and the state.
"""
import numpy as np
from utils.env import environment, envState
from LAC.dataProvider import StaticBatchProvider2D
from utils.graph_utils import init_pixel_graph

class LACenv(environment):
    def __init__(self, staticBatchProvider, options):
        super(LACenv, self).__init__()
        self.staticBatchProvider = staticBatchProvider
        self.options = options


class LACstate(envState):
    def __init__(self,
                 staticBatchProvider,
                 options):
        """

        :param staticBatchProvider:
        :type staticBatchProvider: StaticBatchProvider2D
        :param options:
        """
        super(LACstate, self).__init__()
        self.batchProv = staticBatchProvider
        # TODO: missing options...!
        self.batchProv.init_staticBatch()
        self.options = options

        self._init_SP_graph()
        self._init_UnionFinds()
        self._init_priority_queue()

    def _init_SP_graph(self):
        bs, sizeX, sizeY = self.batchProv.get_predictionLabels_shape()
        self.graphs = []
        for b in range(bs):
            self.graphs.append(init_pixel_graph(
                                   [sizeX, sizeY],
                                    get_node_features=self._init_node_features,
                                    get_edge_features=self._init_edge_features,
                                    selected_batch=b))


    def _init_node_features(self, pixel_grid, selected_batch=None):
        """
        Function used for init_pixel_graph()

        :param pixel_grid:
        :return:
        """
        if not selected_batch:
            raise TypeError("Missing selected batch")

        labelsID = self.batchProv.batchLabels[selected_batch]

        print "THIS IS ORRIBLE..."
        NP = np.product(pixel_grid.shape)
        features = np.empty(NP, dtype=tuple)
        for i, ID in enumerate(labelsID.flatten()):
            features[i] = (i, {'GT_ids': [ID], 'GT_sizes': [1]})

        return features


    def _init_edge_features(self, pixel_grid, edges):
        """
        Function used for init_pixel_graph()

        :param pixel_grid:
        :param edges:
        :return:
        """
        # TODO: to be implemented


    def _init_UnionFinds(self):
        pass

    def _init_priority_queue(self):
        pass




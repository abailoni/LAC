"""
Here we construct the specific environment and the state.
"""
import numpy as np
from utils.env import environment, envState
from LAC.dataProvider import StaticBatchProvider2D
from utils.graph_utils import init_pixel_graph
from utils.extUnionFind import UnionFind
from utils.numpy_utils import DynamicRecArray
from Queue import PriorityQueue
import logging

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

        self.log = logging.getLogger(__name__)

        self.batchProv = staticBatchProvider
        # TODO: missing options...!
        self.batchProv.init_staticBatch()
        self.options = options

        # Initialize data-structure:
        self.bs, sizeX, sizeY = self.batchProv.get_predictionLabels_shape()
        # Shape: (batches, 2, num_pairs)
        self.pixEdges = np.empty((self.bs, 2, sizeX * (sizeX - 1) + sizeY * (sizeY - 1)), dtype=np.int16)

        self._init_UnionFinds()
        self._init_SP_graph()
        self._init_priority_queue()

        history_dtype = [
            ('SP1', np.int16),
            ('SP2', np.int16),
            ('edge_ID', np.int16),
            ('p_merge', np.float32),
            ('Delta_w', np.float32),
            ('update_aff', np.bool_),
            ('merged', np.bool_)]

        self.history = [ DynamicRecArray(history_dtype, resizing_frequency=100) for _ in range(self.bs)]

        # Select the first edges from PQ:
        self._get_next_edges_fromPQ()

        '''
        self.updating_affinities: (type Boolean)
          - False:
              * self.bs selected edges were extracted from the PQ
              * the state requires self.bs actions to be updated
              * merging decisions are performed
          - True:
              * after some merges, N selected edges costs need to be updated
              * the state requires N actions to be updated
              * merging decisions are NOT performed (only update of the PQ)
        '''
        self.updating_affinities = False




    def _init_SP_graph(self):
        bs, sizeX, sizeY = self.batchProv.get_predictionLabels_shape()
        self.graphs = []
        for b in range(bs):
            self.graphs.append(init_pixel_graph(
                                   [sizeX, sizeY],
                                    get_node_features=self._init_node_features,
                                    get_edge_features=self._init_edge_features,
                                    get_edge_weights=self._init_edge_weights,
                                    selected_batch=b))


    def _init_node_features(self, pixel_grid, selected_batch=None):
        """
        Function used for init_pixel_graph()

        :param pixel_grid:
        :return:
        """
        if selected_batch is None:
            raise TypeError("Missing selected batch")

        labelsID = self.batchProv.batchLabels[selected_batch]

        # TODO: find better implementation...? (horrible)
        NP = np.product(pixel_grid.shape)
        features = np.empty(NP, dtype=tuple)
        for i, ID in enumerate(labelsID.flatten()):
            features[i] = (i, {'GT_ids': [ID], 'GT_sizes': [1]})

        return features

    def _init_edge_weights(self, axis=0, selected_batch=None):
        if selected_batch is None:
            raise TypeError("Missing selected batch")

        # TODO: this is not general enough for the 3D batch
        # Order of the affinities: z, x, y
        return self.batchProv.staticBatch[selected_batch, axis + 2, ...]

    def _init_edge_features(self, pixel_grid, edges, edges_weights, selected_batch=None):
        """
        Function used for init_pixel_graph()

        :param pixel_grid:
        :param edges:
        :return:
        """
        if selected_batch is None:
            raise TypeError("Missing selected batch")

        num_edges = edges.shape[0]

        self.pixEdges[selected_batch] = np.transpose(edges)

        bs, sizeX, sizeY = self.batchProv.get_predictionLabels_shape()
        assert (sizeX*(sizeX-1) + sizeY*(sizeY-1) == num_edges)

        # TODO: this is even worse...
        edges_out = np.empty(num_edges, dtype=tuple)
        for i, (u, v, weight) in enumerate(zip(edges[:,0], edges[:,1], edges_weights)):
            edges_out[i] = (u, v, {
                'weight': weight,
                'UF-id': i, # Id linking the edge to self.edges_unFind
                'postponed': False,
                'outdated': False,
                'hiddenState': None,
                'SP-sizes': (1,1)})

        return edges_out


    def _init_UnionFinds(self):
        bs, sizeX, sizeY = self.batchProv.get_predictionLabels_shape()
        self.SP_unFind, self.edges_unFind = [], []
        for b in range(bs):
            self.SP_unFind.append(UnionFind(sizeX*sizeY))
            self.edges_unFind.append(UnionFind(sizeX*(sizeX-1) + sizeY*(sizeY-1)))


    def _init_priority_queue(self):
        self.priority_queue = []
        self.log.debug("Initializing priority queues...")
        for b, edges in enumerate(self.pixEdges):
            q = PriorityQueue()
            # TODO: again slow
            for u, v in zip(edges[0], edges[1]):
                q.put((self.graphs[b][u][v][0]['weight'], self.graphs[b][u][v][0]['UF-id']))
            self.priority_queue.append(q)


    def _get_next_edges_fromPQ(self):
        selected_edges = []
        for b in range(self.bs):
            while True:
                weight, pixelEdge_ID = self.priority_queue[b].get()
                # TODO: should I use the root edge?
                # root_pixelEdge_ID = self.edges_unFind[b].find(pixelEdge_ID)
                pix1, pix2 = self.pixEdges[b, :, pixelEdge_ID]
                node1, node2 = self.SP_unFind[b].find(pix1), self.SP_unFind[b].find(pix2)

                self.log.info((weight,pixelEdge_ID,node1,node2))




                # Check if the edge was merged in the meantime:
                if (node1 not in self.graphs[b]) or (node2 not in self.graphs[b]):
                    continue
                # Check if the affinity value was updated in the meantime:
                elif self.graphs[b][node1][node2][0]['weight'] != weight:
                    self.log.warning(("Updated weight.. Careful, double precision and equal sign...",
                                      weight,
                                      self.graphs[b][node1][node2][0]['weight']))
                    continue
                else:
                    break
            selected_edges.append(pixelEdge_ID)

            # Update history state:
            new_entry = (node1, node2, pixelEdge_ID, -9999., -9999., False, False)
            print new_entry
            self.history[b] = self.history[b].append(new_entry)
            print self.history[b]()
            raise ValueError("This is the END")


        return self._prepare_NetInput(selected_edges)


    def _prepare_NetInput(self, selected_edges):
        # TODO: generalize for arbitrary num. of edges and specific batch indices

        centers = self.get_pred_coords_from_pixEdge(selected_edges)

        # Static glimpse input:
        self.cropped_staticInput = self.batchProv.get_cropped_staticBatch(centers)

        # Dynamic glimpse input:
        pass

        # Dynamic coarse input:
        pass

        return self.render()


    def get_pred_coords_from_pixEdge(self, edges):
        """
        Takes an array of pixelEdge_IDs and return the xy-coords of the center (then fed to
        batchProvider.get_cropped_staticBatch)

        Here I ignore which affinity I extracted (along which direction, it does not change
        the position of the network and anyway the weight is stored in the graph)

        :param edges:
        :return:
        """
        uPixels, vPixels = self.pixEdges[range(self.bs), :, edges].T

        szX, szY = self.batchProv.sizeXYpred
        pixel_grid = np.arange(szX*szY).reshape((szX,szY))

        centers = [np.argwhere(pixel_grid == uPixels[b])[0] for b in range(self.bs)]
        return np.array(centers)


    def update_state(self, actions):
        # In our case we ignore the reward:
        _ = super(LACstate, self).update_state(actions)

        assert(len(actions.shape) == 2)
        assert(actions.shape[1] == 2)

        if self.updating_affinities:
            self._update_affinities(actions)
        else:
            self._perform_actions(actions)

        return self.render()

    def _perform_actions(self, actions):
        assert(actions.shape[0]==self.bs)

        # Update graph and stuff:
        pass

        if True:
            # If merges, output new state for affinities update
            self.updating_affinities = True
            self.update_batchSize = 10
            pass
        else:
            # If not, get new edges from PQ
            self.updating_affinities = False
            self._get_next_edges_fromPQ()

    def _update_affinities(self, actions):
        assert(actions.shape[0] == self.update_batchSize)


        pass


        # Get new edges:
        self.updating_affinities = False
        self._get_next_edges_fromPQ()

    def render(self):
        super(LACstate, self).render()

        # Render whatever, net inputs, etc...
        out = None
        pass


        return self.updating_affinities, out

"""
Some utility functions for dealing with graphs. For the moment using networkx module
"""
import time
import numpy as np
import networkx as nx
import logging

def init_pixel_graph(
        shape,
        n_dims=None,
        get_node_features=None,
        get_edge_features=None,
        **feature_kwargs):
    """
    get_node_features should output something like:

          [(1, {'GTid': 4}),
           (2, {'GTid': 5}),
           ... ]

    with input the pixel grid.


    :param shape: can be an integer or a list of dimensions
    :param n_dims: optional
    :param feature_kwargs: kwargs inputed to the two feature functions

    :return: the created graph
    """
    log = logging.getLogger(__name__)
    if isinstance(shape, (int, long)):
        if not n_dims:
            raise ValueError("Missing number of dimensions")
        N_pixels = shape**n_dims
        shape = np.array([shape]*n_dims)
    else:
        log.warning("Missing check: is shape a list or an array...?")
        shape = np.array(shape)
        N_pixels = np.product(shape)

    pixel_grid = np.arange(N_pixels).reshape(shape)

    # Compute edges for adjacency graph:
    edges = np.ones([n_dims, 2] + list(shape), dtype=np.int32)
    for axis in range(n_dims):
        edges[axis, 0, ...] = pixel_grid
        edges[axis, 1, ...] = np.roll(pixel_grid, -1, axis=axis)
        # Remember about the last one:
        last_col_slice = [axis, 0] + [slice(None)] * axis + [slice(-1, None)] + [slice(None)] * (n_dims - 1 - axis)
        edges[last_col_slice] = -9999

    # Delete redundant edges:
    edges = np.reshape(np.transpose(edges, [0] + range(2, 2 + n_dims) + [1]), [-1, 2])
    indx_redundant = np.nonzero(edges[:, 0] == -9999)
    edges = np.delete(edges, indx_redundant, axis=0)

    # Collect features:
    if get_node_features:
        nodes = get_node_features(pixel_grid, **feature_kwargs)
    else:
        nodes = pixel_grid.flatten()

    if get_edge_features:
        edges = get_edge_features(pixel_grid,edges,**feature_kwargs)

    log.info("Creating graph...")
    tick = time.time()
    G = nx.MultiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    tock = time.time()
    log.debug("Graph created, %f secs" % (tock - tick))

    return G



def contract_nodes(G, u, v, self_loops=False):
    """
    MODIFIED VERSION FROM NETWORKX:
        - the inputed graph is directly modified, no copies created
        - u is kept, v not
        - if the graph is MultiGraph, then all the neighbouring connections are kept



    Returns the graph that results from contracting ``u`` and ``v``.

    Node contraction identifies the two nodes as a single node incident to any
    edge that was incident to the original two nodes.

    Parameters
    ----------
    G : NetworkX graph
       The graph whose nodes will be contracted.

    u, v : nodes
       Must be nodes in ``G``.

    self_loops : Boolean
       If this is ``True``, any edges joining ``u`` and ``v`` in ``G`` become
       self-loops on the new node in the returned graph.

    Returns
    -------
    Networkx graph
       A new graph object of the same type as ``G`` (leaving ``G`` unmodified)
       with ``u`` and ``v`` identified in a single node. The right node ``v``
       will be merged into the node ``u``, so only ``u`` will appear in the
       returned graph.

    Examples
    --------
    Contracting two nonadjacent nodes of the cycle graph on four nodes `C_4`
    yields the path graph (ignoring parallel edges)::


    See also
    --------
    contracted_edge
    quotient_graph

    Notes
    -----
    This function is also available as ``identified_nodes``.
    """
    H = G
    if H.is_directed():
        in_edges = ((w, u, d) for w, x, d in G.in_edges(v, data=True)
                    if self_loops or w != u)
        out_edges = ((u, w, d) for x, w, d in G.out_edges(v, data=True)
                     if self_loops or w != u)
        new_edges = chain(in_edges, out_edges)
    else:
        new_edges = ((u, w, d) for x, w, d in G.edges(v, data=True)
                     if self_loops or w != u)
    v_data = H.node[v]
    H.remove_node(v)
    H.add_edges_from(new_edges)
    if 'contraction' in H.node[u]:
        H.node[u]['contraction'][v] = v_data
    else:
        H.node[u]['contraction'] = {v: v_data}
    return H

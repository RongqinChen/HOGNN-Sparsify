from itertools import combinations, permutations
from typing import List

import networkx as nx
import numpy as np
from networkx import Graph as NX_Graph
from sage.all import Graph as SAGE_Graph
from sage.graphs.connectivity import blocks_and_cuts_tree, spqr_tree


def compute_pairconn_in_spqr_comp(label, spqr_comp: SAGE_Graph, conn_mat):
    nodes = spqr_comp.vertices()
    edges = spqr_comp.edges()
    print(label, nodes, edges)

    # S: the associated graph is a cycle graph with three or more vertices and edges.
    if label == "S":
        for u, v in permutations(nodes, 2):
            conn_mat[u, v] += 2

        for edge in edges:  # undirected
            if edge[2] is not None and 'new' in edge[2]:
                v, u = edge[0], edge[1]
                conn_mat[v, u] -= 1
                conn_mat[u, v] -= 1

    # P: the associated graph is a dipole graph, a multigraph with two vertices and three or more edges.
    # Q: the associated graph has a single real edge.
    elif label == "P" or label == "Q":
        # assert len(nodes) == 2
        for edge in edges:  # undirected
            if edge[2] is None:
                v, u = edge[0], edge[1]
                conn_mat[v, u] += 1
                conn_mat[u, v] += 1

    # R: the associated graph is a 3-connected graph that is not a cycle or dipole.
    elif label == "R":
        nxg = NX_Graph()
        nxg.add_nodes_from(nodes)
        nxg.add_edges_from([(edge[0], edge[1]) for edge in edges])
        ret_dict = nx.connectivity.all_pairs_node_connectivity(nxg)

        # both ret_dict[u][v] and ret_dict[v][u] are included in ret_dict,
        # so we only need to add once
        for u, v_conn in ret_dict.items():
            for v, conn in v_conn.items():
                conn_mat[u, v] += conn

        for edge in edges:  # undirected
            if edge[2] is not None and 'new' in edge[2]:
                v, u = edge[0], edge[1]
                conn_mat[v, u] -= 1
                conn_mat[u, v] -= 1


def compute_pairconn_in_block_comp(block_comp: SAGE_Graph, conn_mat, cnt_dict):
    a_spqr_tree = spqr_tree(block_comp)
    for label, spqr_comp in a_spqr_tree.vertices():
        compute_pairconn_in_spqr_comp(label, spqr_comp, conn_mat)

    if cnt_dict is not None:
        for label, spqr_comp in a_spqr_tree.vertices():
            size = len(spqr_comp.vertices())
            cnt_dict[f"{label}_sizes"].append(size)

    block_nodes = block_comp.vertices()
    for u, v in combinations(block_nodes, 2):
        conn_mat[u, v] = max(conn_mat[u, v], 2)
        conn_mat[v, u] = conn_mat[u, v]


def compute_pair_conn_in_comp(comp: SAGE_Graph, conn_mat, cnt_dict):
    bc_tree = blocks_and_cuts_tree(comp)
    block_comp_list = [
        comp.subgraph(bc[1])
        for bc in bc_tree.vertices()
        if bc[0] == "B" and len(bc[1]) > 2  # block could be a trivial edge graph
    ]
    for block_comp in block_comp_list:
        compute_pairconn_in_block_comp(block_comp, conn_mat, cnt_dict)

    if cnt_dict is not None:
        block_sizes = [len(block.vertices()) for block in block_comp_list]
        cnt_dict["B_sizes"].extend(block_sizes)

    comp_nodes = comp.vertices()
    for u, v in combinations(comp_nodes, 2):
        conn_mat[u, v] = max(conn_mat[u, v], 1)
        conn_mat[v, u] = conn_mat[u, v]


def compute_pair_conn(sage_graph: SAGE_Graph, cnt_dict: dict = None) -> np.ndarray:
    num_nodes = sage_graph.num_verts()
    num_edges = sage_graph.num_edges()

    if cnt_dict is not None:
        cnt_dict.update({
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "C_sizes": [], "B_sizes": [],
            "S_sizes": [], "P_sizes": [],
            "Q_sizes": [], "R_sizes": [],
        })

    conn_mat = np.zeros((num_nodes, num_nodes), dtype=np.int32)

    # Split all connected components in the graph.
    nodes_list: List[List[int]] = sage_graph.connected_components(sort=False)
    comp_list = [sage_graph.subgraph(nodes) for nodes in nodes_list]
    if cnt_dict is not None:
        comp_sizes = [len(comp.vertices()) for comp in comp_list]
        cnt_dict["C_sizes"].extend(comp_sizes)

    # Compute pair connectivity for each connected component.
    for comp in comp_list:
        compute_pair_conn_in_comp(comp, conn_mat, cnt_dict)

    # Set self-loop connectivity for each node
    for v in range(num_nodes):
        conn_mat[v, v] = conn_mat[v, :].max()

    pair_conn = conn_mat.flatten()
    return pair_conn


def test1():
    from .graph_drawio import draw_graph_drawio
    nx_graph = NX_Graph()
    nx_graph.add_nodes_from(range(6))
    nx_graph.add_edges_from([
        (0, 1), (0, 2), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5),
    ])
    pos = nx.kamada_kawai_layout(nx_graph)
    pos = {key: 200 * val for key, val in pos.items()}
    draw_graph_drawio(nx_graph, pos, filepath='images/local_conn_test1.drawio')

    sage_graph = SAGE_Graph(nx_graph)
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    print(pair_conn.shape)
    num_nodes = sage_graph.num_verts()
    for idx in range(num_nodes):
        print(f"{idx:02d}: ", end='')
        for jdx in range(idx + 1):
            print(f"{jdx:02d}={pair_conn[idx * num_nodes + jdx]}", end='  ')
        print()


def test1_1():
    # including backward-edge
    from .graph_drawio import draw_graph_drawio
    nx_graph = NX_Graph()
    nx_graph.add_nodes_from(range(6))
    nx_graph.add_edges_from([
        (0, 1), (0, 2), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5),
        (2, 0)
    ])
    pos = nx.kamada_kawai_layout(nx_graph)
    pos = {key: 200 * val for key, val in pos.items()}
    draw_graph_drawio(nx_graph, pos, filepath='images/local_conn_test1_1.drawio')

    sage_graph = SAGE_Graph(nx_graph)
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    num_nodes = sage_graph.num_verts()
    print(pair_conn.shape)
    for idx in range(num_nodes):
        print(f"{idx:02d}: ", end='')
        for jdx in range(idx + 1):
            print(f"{jdx:02d}={pair_conn[idx * num_nodes + jdx]}", end='  ')
        print()


def test2():
    from .graph_drawio import draw_graph_drawio
    nx_graph = NX_Graph()
    nx_graph.add_nodes_from(range(6))
    nx_graph.add_edges_from([
        (0, 1), (0, 2), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5),
        (0, 4), (1, 2)
    ])
    pos = nx.kamada_kawai_layout(nx_graph)
    pos = {key: 200 * val for key, val in pos.items()}
    draw_graph_drawio(nx_graph, pos, filepath='images/local_conn_test2.drawio')

    sage_graph = SAGE_Graph(nx_graph)
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    num_nodes = sage_graph.num_verts()
    print(pair_conn.shape)
    for idx in range(num_nodes):
        print(f"{idx:02d}: ", end='')
        for jdx in range(idx + 1):
            print(f"{jdx:02d}={pair_conn[idx * num_nodes + jdx]}", end='  ')
        print()


def test3():
    from .graph_drawio import draw_graph_drawio
    nx_graph = NX_Graph()
    nx_graph.add_nodes_from(range(6))
    nx_graph.add_edges_from([
        (0, 1), (0, 2), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5),
        (0, 4), (1, 2), (3, 4)
    ])
    pos = nx.kamada_kawai_layout(nx_graph)
    pos = {key: 200 * val for key, val in pos.items()}
    draw_graph_drawio(nx_graph, pos, filepath='images/local_conn_test3.drawio')

    sage_graph = SAGE_Graph(nx_graph)
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    num_nodes = sage_graph.num_verts()
    print(pair_conn.shape)
    for idx in range(num_nodes):
        print(f"{idx:02d}: ", end='')
        for jdx in range(idx + 1):
            print(f"{jdx:02d}={pair_conn[idx * num_nodes + jdx]}", end='  ')
        print()


def test4():
    from .graph_drawio import draw_graph_drawio
    nx_graph = NX_Graph()
    nx_graph.add_nodes_from(range(8))
    nx_graph.add_edges_from([
        (0, 1), (0, 2), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5),
        (0, 4), (1, 2), (3, 4), (4, 6), (2, 7), (6, 7), (2, 6),
        (4, 7), (2, 5),
    ])
    pos = nx.kamada_kawai_layout(nx_graph)
    pos = {key: 200 * val for key, val in pos.items()}
    draw_graph_drawio(nx_graph, pos, filepath='images/local_conn_test4.drawio')

    sage_graph = SAGE_Graph(nx_graph)
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    num_nodes = sage_graph.num_verts()
    print(pair_conn.shape)
    for idx in range(num_nodes):
        print(f"{idx:02d}: ", end='')
        for jdx in range(idx + 1):
            print(f"{jdx:02d}={pair_conn[idx * num_nodes + jdx]}", end='  ')
        print()


def dodecahedron():
    from .graph_drawio import draw_graph_drawio
    nx_graph = NX_Graph()
    nx_graph.add_nodes_from(range(20))
    nx_graph.add_edges_from([
        (0, 1), (0, 4), (0, 5),
        (1, 2), (1, 6),
        (2, 3), (2, 7),
        (3, 4), (3, 8), (4, 9),
        (5, 12), (5, 13),
        (6, 13), (6, 14),
        (7, 10), (7, 14),
        (8, 10), (8, 11),
        (9, 11), (9, 12),
        (10, 15), (11, 16), (12, 17), (13, 18), (14, 19),
        (15, 16), (15, 19), (16, 17), (17, 18), (18, 19)
    ])
    # pos = nx.spring_layout(nx_graph)

    C0 = np.cos(np.pi/2)
    S0 = -np.sin(np.pi/2)
    C1 = np.cos(np.pi/2-2*np.pi/5) 
    S1 = -np.sin(np.pi/2-2*np.pi/5)
    C2 = np.cos(np.pi/2-2*np.pi/5*2)
    S2 = -np.sin(np.pi/2-2*np.pi/5*2)
    C3 = np.cos(np.pi/2-2*np.pi/5*3)
    S3 = -np.sin(np.pi/2-2*np.pi/5*3)
    C4 = np.cos(np.pi/2-2*np.pi/5*4)
    S4 = -np.sin(np.pi/2-2*np.pi/5*4)

    pos = {
        0: (7*C0, 7*S0), 1: (7*C1, 7*S1), 2: (7*C2, 7*S2), 3: (7*C3, 7*S3), 4: (7*C4, 7*S4),
        5: (5*C0, 5*S0), 6: (5*C1, 5*S1), 7: (5*C2, 5*S2), 8: (5*C3, 5*S3), 9: (5*C4, 5*S4),
        10: (-5*C0, -5*S0), 11: (-5*C1, -5*S1), 12: (-5*C2, -5*S2), 13: (-5*C3, -5*S3), 14: (-5*C4, -5*S4),
        15: (-3*C0, -3*S0), 16: (-3*C1, -3*S1), 17: (-3*C2, -3*S2), 18: (-3*C3, -3*S3), 19: (-3*C4, -3*S4),
    }
    pos = {key: 30 * np.array(val) for key, val in pos.items()}
    print(pos[0])
    draw_graph_drawio(nx_graph, pos, filepath='images/dodecahedron.drawio')

    sage_graph = SAGE_Graph(nx_graph)
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    num_nodes = sage_graph.num_verts()
    print(pair_conn.shape)
    for idx in range(num_nodes):
        print(f"{idx:02d}: ", end='')
        for jdx in range(idx + 1):
            print(f"{jdx:02d}={pair_conn[idx * num_nodes + jdx]}", end='  ')
        print()


def octagon():
    from .graph_drawio import draw_graph_drawio
    nx_graph = NX_Graph()
    nx_graph.add_nodes_from(range(9))
    nx_graph.add_edges_from([
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
        (8, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    ])
    pos = {0: (0, 0)}
    r = 30
    for idx in range(1, 9):
        angle = 2*np.pi/8*(idx-1)
        pos[idx] = (r*np.cos(angle), r*np.sin(angle))
    node_labels = {idx: '' for idx in range(9)}
    draw_graph_drawio(nx_graph, pos, node_labels=node_labels, node_size=10, filepath='images/octagon.drawio')

    sage_graph = SAGE_Graph(nx_graph)
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    num_nodes = sage_graph.num_verts()
    print(pair_conn.shape)
    for idx in range(num_nodes):
        print(f"{idx:02d}: ", end='')
        for jdx in range(idx + 1):
            print(f"{jdx:02d}={pair_conn[idx * num_nodes + jdx]}", end='  ')
        print()


def regular(num_nodes):
    from .graph_drawio import draw_graph_drawio
    nx_graph = NX_Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from([
        (idx, (idx + 1) % num_nodes) for idx in range(num_nodes)
    ])
    pos = {}
    r = 30
    for idx in range(num_nodes):
        angle = 2*np.pi/num_nodes*idx + np.pi / 2
        pos[idx] = (r*np.cos(angle), r*np.sin(angle))
    node_labels = {idx: '' for idx in range(num_nodes)}
    draw_graph_drawio(nx_graph, pos, node_labels=node_labels, node_size=10, filepath=f'images/regular_{num_nodes}.drawio')

    sage_graph = SAGE_Graph(nx_graph)
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    num_nodes = sage_graph.num_verts()
    print(pair_conn.shape)
    for idx in range(num_nodes):
        print(f"{idx:02d}: ", end='')
        for jdx in range(idx + 1):
            print(f"{jdx:02d}={pair_conn[idx * num_nodes + jdx]}", end='  ')
        print()


def x2():
    num_nodes = 9
    from .graph_drawio import draw_graph_drawio
    nx_graph = NX_Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from([
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
        (7, 1), (8, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8),
    ])
    node_labels = None
    r = 30
    pos = {0: (0, 0)}
    for idx in range(1, num_nodes):
        angle = 2*np.pi/(num_nodes-1)*idx + np.pi / 2
        pos[idx] = (r*np.cos(angle), r*np.sin(angle))
    # node_labels = {idx: '' for idx in range(num_nodes)}
    draw_graph_drawio(nx_graph, pos, node_labels=node_labels, node_size=10, filepath=f'images/x2_{num_nodes}.drawio')

    sage_graph = SAGE_Graph(nx_graph)
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    num_nodes = sage_graph.num_verts()
    print(pair_conn.shape)
    for idx in range(num_nodes):
        print(f"{idx:02d}: ", end='')
        for jdx in range(idx + 1):
            print(f"{jdx:02d}={pair_conn[idx * num_nodes + jdx]}", end='  ')
        print()


def x():
    from .graph_drawio import draw_graph_drawio
    nx_graph = NX_Graph()
    nx_graph.add_nodes_from(range(11))
    nx_graph.add_edges_from([
        (0, 1), (0, 6),
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 1)
    ])
    pos = nx.kamada_kawai_layout(nx_graph)
    pos = {key: 200 * val for key, val in pos.items()}
    draw_graph_drawio(nx_graph, pos, filepath='images/x.drawio')

    sage_graph = SAGE_Graph(nx_graph)
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    num_nodes = sage_graph.num_verts()
    print(pair_conn.shape)
    for idx in range(num_nodes):
        print(f"{idx:02d}: ", end='')
        for jdx in range(idx + 1):
            print(f"{jdx:02d}={pair_conn[idx * num_nodes + jdx]}", end='  ')
        print()


def x3():
    from .graph_drawio import draw_graph_drawio
    nx_graph = NX_Graph()
    nx_graph.add_nodes_from(range(8))
    nx_graph.add_edges_from([
        (0, 1), (0, 5), (0, 6), 
        (1, 2), (1, 6), (1, 7),
        (2, 3), (2, 7),
        (3, 4), (3, 7),
        (4, 5), (4, 6), (4, 7),
        (5, 6),
    ])
    pos = nx.kamada_kawai_layout(nx_graph)
    pos = {key: 200 * val for key, val in pos.items()}
    draw_graph_drawio(nx_graph, pos, filepath='images/x3.drawio')

    sage_graph = SAGE_Graph(nx_graph)
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    num_nodes = sage_graph.num_verts()
    print(pair_conn.shape)
    for idx in range(num_nodes):
        print(f"{idx:02d}: ", end='')
        for jdx in range(idx + 1):
            print(f"{jdx:02d}={pair_conn[idx * num_nodes + jdx]}", end='  ')
        print()


if __name__ == "__main__":
    # test1()
    # test1_1()
    # test2()
    # test3()
    # test4()
    # dodecahedron()
    # octagon()
    # regular(6)
    # x2()
    x3()

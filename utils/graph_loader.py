import ray
import igraph as ig
import numpy as np
import os
from utils.graph_generator import label_graph_new, label_graph_new_optimized

from itertools import islice
import multiprocessing as mp
import h5py


def chunked_edge_reader(file_path, chunk_size=10_000_000):
    with open(file_path, 'r') as f:
        while True:
            lines = list(islice(f, chunk_size))
            if not lines:
                break
            yield lines


def process_chunk(chunk):
    edges = []
    for line in chunk:
        try:
            u, v = map(int, line.strip().split())
            edges.append((u, v))
        except ValueError:
            continue
    return list(set(edges))


def load_large_graph_optimized_base(file_path, directed=False):
    graph = ig.Graph(directed=directed)
    vertex_set = set()
    edges = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        processed_chunks = pool.imap(process_chunk, chunked_edge_reader(file_path))
        for chunk_edges in processed_chunks:
            edges.extend(chunk_edges)
            vertex_set.update({u for edge in chunk_edges for u in edge})
    max_node_id = max(vertex_set) if vertex_set else 0
    graph.add_vertices(max_node_id + 1)
    graph.add_edges(edges)
    return graph


def load_large_graph_optimized(file_path, community_path=None, sharpen_boundary=True, saved_hdf5_path=None):
    ray.init(ignore_reinit_error=True)
    if saved_hdf5_path and os.path.exists(saved_hdf5_path):
        print(f"Loading optimized graph from HDF5: {saved_hdf5_path}")
        graph = load_graph_hdf5(saved_hdf5_path)
    else:
        print("No HDF5 file found. Generating graph from raw data...")
        if file_path == "data/com-wiki.preprocessed.ungraph.txt":
            graph = load_large_graph_optimized_base(file_path, directed=True)
        else:
            graph = load_large_graph_optimized_base(file_path)
        if community_path:
            communities = load_communities(community_path)
            label_graph_new_optimized(graph, communities, sharpen_boundary)
        if saved_hdf5_path:
            save_graph_hdf5(graph, saved_hdf5_path)
    graph_ref = ray.put(graph)
    print("Finished loading optimized graph")
    return graph_ref


def load_communities(community_path):
    with open(community_path, 'r') as f:
        communities = []
        for line in f:
            nodes = line.strip().replace('\t', ' ').split()
            if len(nodes) == 0:
                continue
            communities.append(frozenset(map(int, nodes)))
        print(f"Successfully loading {len(communities)} communities，Sizes (example): {[len(c) for c in communities[:3]]}")
        return communities


def save_graph_hdf5(graph, file_path):
    with h5py.File(file_path, 'w') as f:
        edges = graph.get_edgelist()
        f.create_dataset("edges", data=np.array(edges, dtype=np.int32))
        f.create_dataset("community", data=np.array(graph.vs["community"], dtype=np.int32))
        f.create_dataset("isBoundary", data=np.array(graph.es["isBoundary"], dtype=bool))
        f.create_dataset("weight", data=np.array(graph.es["weight"], dtype=np.float32))
        f.attrs["numBoundaryNode"] = graph["numBoundaryNode"]
        f.attrs["numBoundaryEdge"] = graph["numBoundaryEdge"]
        comm_list = list(graph["communities"])
        comm_str = ['\t'.join(map(str, c)) for c in comm_list]
        f.create_dataset("communities", data=comm_str, dtype=h5py.string_dtype())
    print(f"Num of communities: {len(graph['communities'])}, Largest size: {max(len(c) for c in graph['communities'])}")


def load_graph_hdf5(file_path):
    graph = ig.Graph(directed=True)
    with h5py.File(file_path, 'r') as f:
        edges = f["edges"][:]
        graph.add_vertices(np.unique(edges).max() + 1)
        graph.add_edges(edges)
        graph.vs["community"] = f["community"][:]
        graph.es["isBoundary"] = f["isBoundary"][:]
        graph.es["weight"] = f["weight"][:]
        graph["numBoundaryNode"] = f.attrs["numBoundaryNode"]
        graph["numBoundaryEdge"] = f.attrs["numBoundaryEdge"]
        comm_bytes = f["communities"][:]
        comm_str = [s.decode('utf-8') for s in comm_bytes]
        graph["communities"] = {frozenset(map(int, s.split('\t'))) for s in comm_str}
        print("Graph nodes {}, edges {}, communities {}".format(graph.vcount(), graph.ecount(), len(graph["communities"])))
    return graph


def load_facebook(file_path="../data/facebook_combined.txt",
                  community_path="../data/facebook_community.txt",
                  saved_hdf5_path="../data/labeled_facebook.hdf5",
                  sharpen_boundary=True):
    if saved_hdf5_path and os.path.exists(saved_hdf5_path):
        print(f"Loading optimized graph from HDF5: {saved_hdf5_path}")
        graph = load_graph_hdf5(saved_hdf5_path)
    else:
        graph = ig.read(file_path, directed=False)
        louvain_community = np.loadtxt(community_path, dtype="int").tolist()
        communities_temp_dict = {i: set() for i in range(max(louvain_community) + 1)}
        [communities_temp_dict[louvain_community[vertex.index]].add(vertex.index) for vertex in graph.vs]
        communities = {frozenset(c) for c in communities_temp_dict.values()}
        graph.to_directed(mode="mutual")
        graph = label_graph_new(graph, communities, sharpen_boundary=sharpen_boundary)
        if saved_hdf5_path:
            save_graph_hdf5(graph, saved_hdf5_path)
    graph_ref = ray.put(graph)
    return graph_ref


def load_twitter(file_path="../data/twitter_combined.txt",
                 community_path="../data/twitter_community.txt",
                 saved_hdf5_path="../data/labeled_twitter.hdf5",
                 sharpen_boundary=True):
    if saved_hdf5_path and os.path.exists(saved_hdf5_path):
        print(f"Loading optimized graph from HDF5: {saved_hdf5_path}")
        graph = load_graph_hdf5(saved_hdf5_path)
    else:
        print("No HDF5 file found. Generating graph from raw data...")
        graph = ig.Graph.Read_Ncol(file_path, directed=True).simplify()
        louvain_community = np.loadtxt(community_path, dtype="int").tolist()
        communities_temp_dict = {i: set() for i in range(max(louvain_community) + 1)}
        [communities_temp_dict[louvain_community[vertex.index]].add(vertex.index) for vertex in graph.vs]
        communities = {frozenset(c) for c in communities_temp_dict.values()}
        ig.Graph.reverse_edges(graph)
        graph = label_graph_new(graph, communities, sharpen_boundary=sharpen_boundary)
        if saved_hdf5_path:
            save_graph_hdf5(graph, saved_hdf5_path)
    graph_ref = ray.put(graph)
    print("Finished loading optimized graph")
    return graph_ref

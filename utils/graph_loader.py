import igraph as ig
import numpy as np
from utils.graph_generator import label_graph


def load_facebook(file="../data/facebook_combined.txt", community_file="../data/facebook_community.txt"):
    graph = ig.read(file, directed=False)
    louvain_community = np.loadtxt(community_file, dtype="int").tolist()
    communities_temp_dict = {i: set() for i in range(max(louvain_community) + 1)}
    [communities_temp_dict[louvain_community[vertex.index]].add(vertex.index) for vertex in graph.vs]
    communities = {frozenset(c) for c in communities_temp_dict.values()}
    graph.to_directed(mode="mutual")
    graph = label_graph(graph, communities, sharpen_boundary=True)
    return graph


def load_twitter(file="../data/twitter_combined.txt", community_file="../data/twitter_community.txt"):
    graph = ig.Graph.Read_Ncol(file, directed=True).simplify()
    louvain_community = np.loadtxt(community_file, dtype="int").tolist()
    communities_temp_dict = {i: set() for i in range(max(louvain_community) + 1)}
    [communities_temp_dict[louvain_community[vertex.index]].add(vertex.index) for vertex in graph.vs]
    communities = {frozenset(c) for c in communities_temp_dict.values()}
    ig.Graph.reverse_edges(graph)
    graph = label_graph(graph, communities, sharpen_boundary=True)
    return graph


def load_email_Eu(file="../data/email-Eu-core.txt", community_file="../data/email-Eu-core-department-labels.txt"):
    graph = ig.read(file, directed=True)
    community = np.loadtxt(community_file, dtype="int").tolist()
    communities_temp_dict = {i: set() for i in range(42)}
    [communities_temp_dict[community[vertex.index][1]].add(vertex.index) for vertex in graph.vs]
    communities = {frozenset(c) for c in communities_temp_dict.values()}
    graph = label_graph(graph, communities, sharpen_boundary=True)
    return graph

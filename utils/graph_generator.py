import igraph as ig
from networkx.generators.community import LFR_benchmark_graph


def import_nx_network(net):
    graph = ig.Graph(n=net.number_of_nodes(), directed=False)
    graph.add_edges(net.edges())
    graph.to_directed(mode="mutual")
    return graph


def generate_lfr_graph(N, tau1, tau2, mu, **kwargs):
    G = LFR_benchmark_graph(N, tau1, tau2, mu, **kwargs)
    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    graph = import_nx_network(G)
    label_graph(graph, communities)
    return graph


def label_graph(graph, communities, sharpen_boundary=False):
    # label weight for edges
    for i in range(len(graph.es)):
        graph.es[i]["weight"] = 1 / graph.vs[graph.es[i].target].indegree()

    # label community for nodes
    i = 0
    for community in communities:
        for node in community:
            graph.vs[node]["community"] = i
        i = i + 1

    # label isBoundary for nodes and edges
    countBoundaryNode = 0
    countBoundaryEdge = 0
    for node in graph.vs:
        node_in_neighbour = graph.neighbors(node.index, mode="in")
        isBoundaryNode = False
        for u in node_in_neighbour:
            if graph.vs[u]["community"] != graph.vs[node.index]["community"]:
                isBoundaryNode = True
                graph.es.find(_source=u, _target=node.index)["isBoundary"] = True
                countBoundaryEdge += 1
                if sharpen_boundary:
                    w = graph.es.find(_source=u, _target=node.index)["weight"]
                    edge_id = graph.es.find(_source=u, _target=node.index).index
                    graph.es[edge_id]["weight"] = w * 0.1
            else:
                graph.es.find(_source=u, _target=node.index)["isBoundary"] = False
        graph.vs[node.index]["isBoundary"] = isBoundaryNode
        if isBoundaryNode:
            countBoundaryNode += 1

    # label numBoundaryNode and numBoundaryEdge for graph
    graph["numBoundaryNode"] = countBoundaryNode
    graph["numBoundaryEdge"] = countBoundaryEdge

    # label communities for graph
    graph["communities"] = communities
    return graph

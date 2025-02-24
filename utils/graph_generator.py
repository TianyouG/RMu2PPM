import numpy as np


def label_graph_new(graph, communities, sharpen_boundary=True):
    communities = list(communities)
    indegrees = graph.degree(mode="in")
    for edge in graph.es:
        edge["weight"] = 1 / indegrees[edge.target]
    community_dict = {}
    for i, community in enumerate(communities):
        for node in community:
            community_dict[node] = i
    graph.vs["community"] = [community_dict.get(node.index, -1) for node in graph.vs]
    neighbors_dict = {v.index: graph.neighbors(v.index, mode="in") for v in graph.vs}
    edges_to_update = []
    countBoundaryNode = 0
    countBoundaryEdge = 0
    for node in graph.vs:
        isBoundaryNode = False
        for u in neighbors_dict[node.index]:
            if graph.vs[u]["community"] != graph.vs[node.index]["community"]:
                isBoundaryNode = True
                edges_to_update.append((u, node.index))
                countBoundaryEdge += 1
        graph.vs[node.index]["isBoundary"] = isBoundaryNode
        if isBoundaryNode:
            countBoundaryNode += 1
    edge_ids = graph.get_eids(pairs=edges_to_update, error=False)
    for edge_id in edge_ids:
        graph.es[edge_id]["isBoundary"] = True
        if sharpen_boundary:
            graph.es[edge_id]["weight"] *= 1 / np.log2(countBoundaryEdge + 1)
    graph["numBoundaryNode"] = countBoundaryNode
    graph["numBoundaryEdge"] = countBoundaryEdge
    graph["communities"] = communities
    return graph


def label_graph_new_optimized(graph, communities, sharpen_boundary=True):
    if not graph.is_directed():
        graph.to_directed(mode="mutual")
    community_array = np.full(graph.vcount(), -1, dtype=np.int32)
    communities = list(communities)
    all_nodes = set()
    for i, comm in enumerate(communities):
        comm_nodes = list(comm)
        for node in comm_nodes:
            if node < 0 or node >= graph.vcount():
                raise ValueError(f"Node {node} exceeds the range of [0, {graph.vcount() - 1}]")
        all_nodes.update(comm_nodes)
        community_array[comm_nodes] = i
    graph.vs["community"] = community_array.tolist()
    sources = np.array([edge.source for edge in graph.es])
    targets = np.array([edge.target for edge in graph.es])
    source_communities = np.array(graph.vs[sources]["community"])
    target_communities = np.array(graph.vs[targets]["community"])
    is_boundary = (source_communities != target_communities)
    graph.es["isBoundary"] = is_boundary
    is_boundary_node = np.zeros(graph.vcount(), dtype=bool)
    is_boundary_node[targets[is_boundary]] = True
    graph.vs["isBoundary"] = is_boundary_node
    if sharpen_boundary:
        in_degrees = np.array(graph.degree(mode="in"), dtype=np.float32)
        weights = 1.0 / np.where(in_degrees[targets] > 0, in_degrees[targets], 1.0)
        weights[is_boundary] *= 1 / np.log2(np.sum(graph.es["isBoundary"]) + 1)
        graph.es["weight"] = weights
    graph["numBoundaryNode"] = np.sum(graph.vs["isBoundary"])
    graph["numBoundaryEdge"] = np.sum(graph.es["isBoundary"])
    graph["communities"] = communities
    return graph

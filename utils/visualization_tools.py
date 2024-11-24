from typing import Any
import numpy as np
import igraph


def estimate_revenue_2pp(graph: igraph.Graph, communities: set[frozenset[int]], seed_set: set[int],
                         setR: int or None = None, P1: float = 0.8, P2: float = 1.0) -> tuple[Any, Any, Any]:
    expected_revenue_p1 = np.ones(len(communities))
    expected_revenue_p2 = np.ones(len(communities))
    for i in range(setR):
        revenue_p1, revenue_p2 = calculate_revenue_p1_p2(graph, communities, seed_set, P1, P2)
        expected_revenue_p1 += np.array(revenue_p1)
        expected_revenue_p2 += np.array(revenue_p2)
        if i % 10 == 0:
            print("finished {:.2%}".format(i/setR))
    community_sizes = np.array([len(community) for community in communities])
    expected_revenue_p1 = expected_revenue_p1 / community_sizes / setR
    expected_revenue_p2 = expected_revenue_p2 / community_sizes / setR
    expected_revenue = expected_revenue_p1 + expected_revenue_p2
    return expected_revenue_p1, expected_revenue_p2, expected_revenue


def calculate_revenue_p1_p2(graph: igraph.Graph, communities: set[frozenset[int]], seed_set: set[int],
                            P1: float = 0.8, P2: float = 1.0) -> tuple[list[int], list[int]]:
    new_active = seed_set.copy()
    activated = seed_set.copy()
    while new_active:
        next_round_new_active = set()
        for node in new_active.copy():
            edges = graph.es.select(_source=node)
            for edge in edges:
                if edge.target not in activated:
                    is_success = (np.random.uniform(0, 1) < edge["weight"])
                    if is_success:
                        activated.add(edge.target)
                        next_round_new_active.add(edge.target)
        new_active = next_round_new_active
    revenue_p1 = [0 for _ in range(len(communities))]
    revenue_p2 = [0 for _ in range(len(communities))]
    i = 0
    for community in communities:
        community_size = len(community)
        activated_node_size = len(community & activated)
        revenue_p1[i] += P1 * activated_node_size
        revenue_p2[i] += P2 * (community_size - activated_node_size) * activated_node_size / community_size
        i += 1
    return revenue_p1, revenue_p2

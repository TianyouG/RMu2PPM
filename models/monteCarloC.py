import igraph
import numpy as np
import math

P1 = 0.8
P2 = 1.0
GENERALIZED = False
STEEPNESS = 5
MIDPOINT = 0.5


def estimate_revenue(graph: igraph.Graph, communities: set[frozenset[int]], seed_set: set[int],
                     epsilon: float, delta: float, isCBGA: bool = False, usingSA: str = "n",
                     setR: int or None = None) -> float:
    """
    Estimate the revenue under two-phase diffusion model using MC-greedy
    :param graph: directed igraph
    :param communities: the set of communities that contains all nodes in each community
    :param seed_set: the given seed set
    :param epsilon: the epsilon value
    :param delta: the delta value
    :param isCBGA: whether to use CBGA or not, defaults to False
    :param usingSA: whether to use sandwich approximation framework or not, optional "n", "u", or "l", defaults to "n"
    :param setR: the number of Monte Carlo simulations, defaults to None
    :return: the estimated revenue of current seed set; if isCBGA = True, it will only calculate the sum of revenue from
    the item in parameter "communities"
    """
    revenue = []
    if not isCBGA:
        # number of cycles of Monte Carlo
        if setR is None:
            R = int(3 * len(graph.vs) * math.log(2 / delta) / (epsilon * epsilon * len(seed_set) * P1)) + 1
        else:
            R = min(setR, int(3 * len(graph.vs) * math.log(2 / delta) / (epsilon * epsilon * len(seed_set) * P1)) + 1)
        for i in range(R):
            revenue.append(calculate_revenue_ic2ppm(graph, communities, seed_set, isCBGA, usingSA))
    else:
        if setR is None:
            R = int(3 * len(list(communities)[0]) * math.log(2 / delta) / (epsilon * epsilon * len(seed_set) * P1)) + 1
        else:
            R = min(setR, int(3 * len(list(communities)[0]) * math.log(2 / delta) / (
                    epsilon * epsilon * len(seed_set) * P1)) + 1)
        for i in range(R):
            revenue.append(calculate_revenue_ic2ppm(graph, communities, seed_set, isCBGA, usingSA))
    return np.mean(revenue)


def calculate_revenue_ic2ppm(graph: igraph.Graph, communities: set[frozenset[int]], seed_set: set[int],
                             isCBGA: bool = False, usingSA: str = "n") -> float:
    """
    Implementation one round of the IC model
    :param graph: directed igraph
    :param communities: the set of communities that contains all nodes in each community
    :param seed_set: the given seed set
    :param isCBGA: whether to use CBGA or not
    :param usingSA: whether to use sandwich approximation framework or not, optional "n", "u", or "l", defaults to "n"
    :return: the revenue
    """
    new_active = seed_set.copy()
    activated = seed_set.copy()
    while new_active:
        next_round_new_active = set()
        for node in new_active.copy():
            edges = graph.es.select(_source=node)
            for edge in edges:
                if isCBGA and edge["isBoundary"]:
                    continue
                if edge.target not in activated:
                    is_success = (np.random.uniform(0, 1) < edge["weight"])
                    if is_success:
                        activated.add(edge.target)
                        next_round_new_active.add(edge.target)
        new_active = next_round_new_active
    revenue = 0
    if usingSA == "n":  # the original problem
        revenue += P1 * len(activated)
        for community in communities:
            community_size = len(community)
            activated_node_size = len(community & activated)
            if not GENERALIZED:
                revenue += P2 * (community_size - activated_node_size) * activated_node_size / community_size
            else:
                revenue += P2 * adjusted_sigmoid(activated_node_size / community_size)5
    elif usingSA == "u":  # the upper bound problem
        for community in communities:
            community_size = len(community)
            activated_node_size = len(community & activated)
            if activated_node_size > community_size * (P1 + P2) / (P2 * 2):
                revenue += (P1 + P2) * (P1 + P2) * community_size / (4 * P2)
            else:
                revenue += activated_node_size * (P1 + P2 - P2 * activated_node_size / community_size)
    elif usingSA == "l":  # the lower bound problem
        for community in communities:
            community_size = len(community)
            activated_node_size = len(community & activated)
            if activated_node_size > community_size * P1 / P2:
                revenue += P1 * community_size
            else:
                revenue += activated_node_size * (P1 + P2 - P2 * activated_node_size / community_size)
    return revenue


def adjusted_sigmoid(x, steepness=STEEPNESS, midpoint=MIDPOINT):
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))
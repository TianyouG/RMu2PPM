import igraph

from models.general_greedy import greedy
from models.CBGA import community_based_greedy
from models.monteCarloC import estimate_revenue
from models.CELF_greedy import greedy_celf_plus
from models.CELF_CBGA import cbga_celf_plus
from utils.timer import timing_decorator


@timing_decorator
def sandwich_approximation(graph: igraph.Graph, communities: set[frozenset[int]], k: int, epsilon: float, delta: float,
                           isCBGA: bool = False, celf: bool = False, setR: int or None = None) -> set[int]:
    if not isCBGA and not celf:
        seed_set_n, rtime_n = greedy(graph, communities, k, epsilon, delta, usingSA="n", setR=setR)
        seed_set_u, rtime_u = greedy(graph, communities, k, epsilon, delta, usingSA="u", setR=setR)
        seed_set_l, rtime_l = greedy(graph, communities, k, epsilon, delta, usingSA="l", setR=setR)
    elif not isCBGA and celf:
        seed_set_n, rtime_n = greedy_celf_plus(graph, communities, k, epsilon, delta, usingSA="n", setR=setR)
        seed_set_u, rtime_u = greedy_celf_plus(graph, communities, k, epsilon, delta, usingSA="u", setR=setR)
        seed_set_l, rtime_l = greedy_celf_plus(graph, communities, k, epsilon, delta, usingSA="l", setR=setR)
    elif isCBGA and not celf:
        seed_set_n, rtime_n = community_based_greedy(graph, communities, k, epsilon, delta, usingSA="n", setR=setR)
        seed_set_u, rtime_u = community_based_greedy(graph, communities, k, epsilon, delta, usingSA="u", setR=setR)
        seed_set_l, rtime_l = community_based_greedy(graph, communities, k, epsilon, delta, usingSA="l", setR=setR)
    else:
        seed_set_n, rtime_n = cbga_celf_plus(graph, communities, k, epsilon, delta, usingSA="n", setR=setR)
        seed_set_u, rtime_u = cbga_celf_plus(graph, communities, k, epsilon, delta, usingSA="u", setR=setR)
        seed_set_l, rtime_l = cbga_celf_plus(graph, communities, k, epsilon, delta, usingSA="l", setR=setR)
    revenue_n = estimate_revenue(graph, communities, seed_set_n, epsilon, delta, isCBGA, usingSA="n", setR=setR)
    revenue_u = estimate_revenue(graph, communities, seed_set_u, epsilon, delta, isCBGA, usingSA="n", setR=setR)
    revenue_l = estimate_revenue(graph, communities, seed_set_l, epsilon, delta, isCBGA, usingSA="n", setR=setR)
    if revenue_n >= revenue_u and revenue_n >= revenue_l:
        return seed_set_n
    elif revenue_u >= revenue_l:
        return seed_set_u
    else:
        return seed_set_l

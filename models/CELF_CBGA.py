import igraph
import logging
import datetime
import time
from models.monteCarloC import estimate_revenue
from utils.timer import timing_decorator

LOG_FILENAME = "{}.log".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)


@timing_decorator
def cbga_celf_plus(graph: igraph.Graph, communities: set[frozenset[int]], k: int, epsilon: float, delta: float,
                   usingSA: str = "n", setR: int or None = None) -> set[int]:
    start_time = time.time()
    logging.info(
        "Calling function cbga_celf_plus with N={}, k={}, epsilon={}, delta={}, usingSA={}".format(
            len(graph.vs), k, epsilon, delta, usingSA))
    record_seeds = []
    record_time = []
    seed_set = {key: set() for key in range(len(communities))}
    seed_set_size = 0
    # build a dict of community, int: frozenset[int]
    community_index = {graph.vs[list(community)[0]]["community"]: community for community in communities}
    Q = list()
    last_seed = None
    cur_best = None
    for node in graph.vs:
        community_temp = set()
        community_temp.add(community_index[node["community"]])
        u_mg1 = estimate_revenue(graph, community_temp, {node.index}, epsilon, delta, isCBGA=True, usingSA=usingSA,
                                 setR=setR)
        u_prev_best = cur_best
        u_mg2 = u_mg1
        u_flag = 0
        Q.append([node.index, u_mg1, u_prev_best, u_mg2, u_flag])
        cur_best = max(Q, key=lambda q: q[1])[0]
    while seed_set_size < k:
        u = max(Q, key=lambda q: q[1])
        if u[1] < 0:
            break
        if u[4] == seed_set_size:
            u_index = u[0]
            seed_set[graph.vs[u_index]["community"]].add(u_index)
            seed_set_size += 1
            Q = [q for q in Q if q[0] != u_index]
            last_seed = u_index
            record_seeds.append(u_index)
            current_time = time.time()
            elapsed_time = current_time - start_time
            record_time.append(round(elapsed_time, 1))
            logging.info("add node {} to seed set, {:.1f} minutes elapsed, finished adding {:.2%} seeds.".format(u_index, elapsed_time / 60, seed_set_size / k))
            continue
        elif u[2] == last_seed:
            u_index = next(i for i, q in enumerate(Q) if q[0] == u[0])
            Q[u_index][1] = u[3]
        else:
            node = graph.vs[u[0]]
            community_temp = set()
            community_temp.add(community_index[node["community"]])
            S = seed_set[node["community"]]
            if len(S) >= 1:
                u_mg1 = (estimate_revenue(graph, community_temp, S | {u[0]}, epsilon, delta, isCBGA=True, usingSA=usingSA, setR=setR)
                         - estimate_revenue(graph, community_temp, S, epsilon, delta, isCBGA=True, usingSA=usingSA, setR=setR))
            else:
                u_mg1 = estimate_revenue(graph, community_temp, S | {u[0]}, epsilon, delta, isCBGA=True, usingSA=usingSA, setR=setR)
            if u[0] != cur_best:
                if node["community"] == graph.vs[cur_best]["community"]:
                    u_mg2 = (estimate_revenue(graph, community_temp, S | {u[0], cur_best}, epsilon, delta,
                                              isCBGA=True, usingSA=usingSA, setR=setR)
                             - estimate_revenue(graph, community_temp, S | {cur_best}, epsilon, delta, isCBGA=True,
                                                usingSA=usingSA, setR=setR))
                else:
                    u_mg2 = u_mg1
            else:
                u_mg2 = 0
            u_index = next(i for i, q in enumerate(Q) if q[0] == u[0])
            Q[u_index][1] = u_mg1
            Q[u_index][2] = cur_best
            Q[u_index][3] = u_mg2
        Q[u_index][4] = seed_set_size
        cur_best = max(Q, key=lambda q: q[1])[0]
    # Merge the values in the dictionary seed_set into one large set, and output the result.
    seed_set = {node for sub_seed_set in seed_set.values() for node in sub_seed_set}
    # calculate the real revenue for each seed set
    record_revenue = []
    temp_seed_set = set()
    for node in record_seeds:
        temp_seed_set.add(node)
        revenue = estimate_revenue(graph, communities, temp_seed_set, epsilon, delta, isCBGA=False, usingSA="n", setR=setR)
        record_revenue.append(round(revenue, 2))
    logging.info("---------SUMMARY---------")
    logging.info("seeds: " + str(record_seeds))
    logging.info("running time: " + str(record_time))
    logging.info("objective: " + str(record_revenue))
    return seed_set

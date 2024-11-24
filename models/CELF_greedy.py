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
def greedy_celf_plus(graph: igraph.Graph, communities: set[frozenset[int]], k: int, epsilon: float, delta: float,
                     usingSA: str = "n", setR: int or None = None) -> set[int]:
    start_time = time.time()
    logging.info(
        "Calling function greedy_celf_plus with N={}, k={}, epsilon={}, delta={}, usingSA={}".format(
            len(graph.vs), k, epsilon, delta, usingSA))
    record_seeds = []
    record_time = []
    seed_set = set()
    Q = list()
    last_seed = None
    cur_best = None
    count_test = 0
    for node in graph.vs:
        u_mg1 = estimate_revenue(graph, communities, {node.index}, epsilon, delta, usingSA=usingSA, setR=setR)
        u_prev_best = cur_best
        u_mg2 = u_mg1
        u_flag = 0
        Q.append([node.index, u_mg1, u_prev_best, u_mg2, u_flag])
        cur_best = max(Q, key=lambda q: q[1])[0]
        count_test += 1
        print("finished {} nodes in the first for loop".format(count_test))
    while len(seed_set) < k:
        u = max(Q, key=lambda q: q[1])
        if u[1] <= 0:
            break
        if u[4] == len(seed_set):
            u_index = u[0]
            seed_set.add(u_index)
            Q = [q for q in Q if q[0] != u_index]
            last_seed = u_index
            record_seeds.append(u_index)
            current_time = time.time()
            elapsed_time = current_time - start_time
            record_time.append(round(elapsed_time, 1))
            logging.info("add node {} to seed set, {:.1f} minutes elapsed, finished adding {:.2%} seeds.".format(u_index, elapsed_time / 60, len(seed_set) / k))
            continue
        elif u[2] == last_seed:
            u_index = next(i for i, q in enumerate(Q) if q[0] == u[0])
            Q[u_index][1] = u[3]
        else:
            u_mg1 = (estimate_revenue(graph, communities, seed_set | {u[0]}, epsilon, delta, usingSA=usingSA, setR=setR)
                     - estimate_revenue(graph, communities, seed_set, epsilon, delta, usingSA=usingSA, setR=setR))
            if u[0] != cur_best:
                u_mg2 = (estimate_revenue(graph, communities, seed_set | {u[0], cur_best}, epsilon, delta, usingSA=usingSA,
                                          setR=setR)
                         - estimate_revenue(graph, communities, seed_set | {cur_best}, epsilon, delta, usingSA=usingSA,
                                            setR=setR))
            else:
                u_mg2 = 0
            u_index = next(i for i, q in enumerate(Q) if q[0] == u[0])
            Q[u_index][1] = u_mg1
            Q[u_index][2] = cur_best
            Q[u_index][3] = u_mg2
        Q[u_index][4] = len(seed_set)
        cur_best = max(Q, key=lambda q: q[1])[0]
    record_revenue = []
    temp_seed_set = set()
    for node in record_seeds:
        temp_seed_set.add(node)
        revenue = estimate_revenue(graph, communities, temp_seed_set, epsilon, delta, usingSA="n", setR=setR)
        record_revenue.append(round(revenue, 2))
    logging.info("---------SUMMARY---------")
    logging.info("seeds: " + str(record_seeds))
    logging.info("running time: " + str(record_time))
    logging.info("objective: " + str(record_revenue))
    return seed_set

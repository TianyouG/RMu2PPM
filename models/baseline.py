import igraph
import logging
import datetime
import random
import numpy as np
from models.monteCarloC import estimate_revenue
from utils.timer import timing_decorator

LOG_FILENAME = "{}.log".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)


@timing_decorator
def rand(graph: igraph.Graph, communities: set[frozenset[int]], k: int, epsilon: float, delta: float,
         setR: int or None = None, repeat_times: int = 100) -> set[int]:
    logging.info("Calling function rand with N={}, k={}, epsilon={}, delta={}".format(len(graph.vs), k, epsilon, delta))
    res_seed_set = set()
    for _k in range(k):
        record_seed_set = list()
        record_revenue = list()
        for i in range(repeat_times):
            seed_set = set(random.sample(range(len(graph.vs)), _k+1))
            revenue = estimate_revenue(graph, communities, seed_set=seed_set, epsilon=epsilon, delta=delta, setR=setR)
            record_seed_set.append(seed_set)
            record_revenue.append(revenue)
            if i == 0 and _k == k:
                res_seed_set = seed_set
        avg_revenue = np.mean(record_revenue)
        logging.info("---------SUMMARY---------")
        logging.info("k = {}; average revenue: {}".format(_k, round(avg_revenue, 1)))
        [logging.info("objective: {}; seeds: {}".format(round(record_revenue[j], 2), str(list(record_seed_set[j])))) for j in range(repeat_times)]
    logging.info("Reminder: the approximate running time for one round needs to be divided by {} based on the following running time".format(repeat_times))
    return res_seed_set


@timing_decorator
def high_degree(graph: igraph.Graph, communities: set[frozenset[int]], k: int, epsilon: float, delta: float,
                setR: int or None = None, mode: str or None = None) -> set[int]:
    logging.info(
        "Calling function high_degree with N={}, k={}, epsilon={}, delta={}, mode={}".format(len(graph.vs), k, epsilon,
                                                                                             delta, mode))
    if mode is None:
        degrees = graph.degree()
    else:
        degrees = graph.degree(mode=mode)
    record_revenue = list()
    index_degree_pairs = [(i, degrees[i]) for i in range(len(degrees))]
    descending_index_degree_pairs = sorted(index_degree_pairs, key=lambda x: x[1], reverse=True)
    seed_set = set([item[0] for item in descending_index_degree_pairs[:k]])
    seed_set_degrees = [item[1] for item in descending_index_degree_pairs[:k]]
    for _k in range(k):
        _seed_set = set([item[0] for item in descending_index_degree_pairs[:_k+1]])
        revenue = estimate_revenue(graph, communities, seed_set=_seed_set, epsilon=epsilon, delta=delta, setR=setR)
        record_revenue.append(round(revenue, 2))
    logging.info("---------SUMMARY---------")
    logging.info("seeds: " + str(list(seed_set)))
    logging.info("degrees: " + str(seed_set_degrees))
    logging.info("objective: " + str(record_revenue))
    return seed_set

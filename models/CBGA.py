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
def community_based_greedy(graph: igraph.Graph, communities: set[frozenset[int]], k: int, epsilon: float,
                           delta: float, usingSA: str = "n", setR: int or None = None) -> set[int]:
    start_time = time.time()
    logging.info(
        "Calling function community_based_greedy with N={}, k={}, epsilon={}, delta={}, usingSA={}".format(
            len(graph.vs), k, epsilon, delta, usingSA))
    seed_set = {key: set() for key in range(len(communities))}
    current_revenue = {key: 0 for key in range(len(communities))}
    seed_set_size = 0
    # build a dict of community, int: frozenset[int]
    community_index = {graph.vs[list(community)[0]]["community"]: community for community in communities}
    loop_counter = 0
    total_loop_num = k * (len(graph.vs) * 2 - k + 1) / 2
    record_seeds = []
    record_time = []
    while seed_set_size < k:
        max_marginal_revenue = 0
        new_seed = None
        for node in graph.vs:
            if node.index not in seed_set[node["community"]]:
                S = seed_set[node["community"]] | {node.index}
                # print("trying node {} to seed set".format(node.index))
                # logging.info("trying node {} to seed set".format(node.index))
                community_temp = set()
                community_temp.add(community_index[node["community"]])
                marginal_revenue = (estimate_revenue(graph, communities=community_temp, seed_set=S, epsilon=epsilon,
                                                     delta=delta, isCBGA=True, usingSA=usingSA, setR=setR)
                                    - current_revenue[node["community"]])
                if marginal_revenue > max_marginal_revenue:
                    max_marginal_revenue = marginal_revenue
                    new_seed = node.index
            loop_counter += 1
            if loop_counter % 100 == 0:
                current_time = time.time()
                elapsed_time = current_time - start_time
                logging.info(
                    "Processed {:.2%}, elapsed {:.0f} minutes, require an additional {:.0f} minutes to complete".format(
                        loop_counter / total_loop_num, elapsed_time / 60,
                        (total_loop_num - loop_counter) * elapsed_time / (60 * loop_counter)))
        if new_seed is not None:
            seed_set[graph.vs[new_seed]["community"]].add(new_seed)
            seed_set_size += 1
            current_revenue[graph.vs[new_seed]["community"]] += max_marginal_revenue
            current_time = time.time()
            elapsed_time = current_time - start_time
            print("add node {} to seed set, {} minutes elapsed.".format(new_seed, elapsed_time / 60))
            logging.info("add node {} to seed set, {} minutes elapsed.".format(new_seed, elapsed_time / 60))
            record_seeds.append(new_seed)
            record_time.append(round(elapsed_time, 1))
        else:
            break
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
    logging.info("seeds: "+str(record_seeds))
    logging.info("running time: "+str(record_time))
    logging.info("objective: "+str(record_revenue))
    return seed_set

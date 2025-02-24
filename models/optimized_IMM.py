import itertools
import logging
import multiprocessing as mp
import multiprocessing.shared_memory as shm
from collections import defaultdict

import numpy as np
import os
import time
import math
import sys

from models.monteCarloC import estimate_revenue, P1


num_checkpoints = 10
MIN_BATCH_SIZE = 2**10
MAX_BATCH_SIZE = 2**13


def parallel_generate_rr_sets(graph_shared, theta, num_processes=max(20, os.cpu_count())):
    logging.info("Start sampling {} rr-sets.".format(theta))
    with mp.Pool(num_processes) as pool:
        batch_size = min(MAX_BATCH_SIZE, max(MIN_BATCH_SIZE, theta // (2 * num_processes)))
        def task_generator():
            for i in range(theta):
                yield (graph_shared,)
        task_iter = iter(task_generator())
        results = []
        total_processed = 0
        next_checkpoint_idx = 0
        progress_checkpoints = [int(theta * i / num_checkpoints) for i in range(1, num_checkpoints + 1)]
        while total_processed < theta:
            batch = list(itertools.islice(task_iter, max(0, min(batch_size, theta - total_processed))))
            if not batch:
                break
            batch_results = pool.starmap(_sampling_worker, batch)
            results.extend(batch_results)
            total_processed += len(batch)
            if next_checkpoint_idx < len(progress_checkpoints) and total_processed >= progress_checkpoints[next_checkpoint_idx]:
                if theta >= 1000000:
                    logging.info(f"Progress: {total_processed}/{theta} RR sets ({total_processed / theta:.0%} completed)")
                    sys.stdout.flush()
                next_checkpoint_idx += 1
            del batch
    return results


def _sampling_worker(graph_shared):
    sources_shm = shm.SharedMemory(name=graph_shared["sources_shm"])
    targets_shm = shm.SharedMemory(name=graph_shared["targets_shm"])
    weights_shm = shm.SharedMemory(name=graph_shared["weights_shm"])
    communities_shm = shm.SharedMemory(name=graph_shared["communities_shm"])
    sources = np.ndarray((sources_shm.size // np.dtype(np.int32).itemsize,), dtype=np.int32, buffer=sources_shm.buf)
    targets = np.ndarray((targets_shm.size // np.dtype(np.int32).itemsize,), dtype=np.int32, buffer=targets_shm.buf)
    weights = np.ndarray((weights_shm.size // np.dtype(np.float32).itemsize,), dtype=np.float32, buffer=weights_shm.buf)
    communities = np.ndarray((communities_shm.size // np.dtype(np.int32).itemsize,), dtype=np.int32, buffer=communities_shm.buf)

    rng = np.random.default_rng(int.from_bytes(os.urandom(8), 'big'))
    n_nodes = communities.shape[0]
    activated = np.zeros(n_nodes, dtype=bool)
    v = rng.integers(0, n_nodes)
    activated[v] = True
    active = np.zeros(n_nodes, dtype=bool)
    active[v] = True

    while np.any(active):
        batch_nodes = np.where(active)[0]
        active.fill(False)
        edge_mask = np.isin(targets, batch_nodes)
        valid_edges = np.where(edge_mask)[0]
        valid_count = len(valid_edges)
        if valid_count == 0:
            continue
        valid_u = sources[valid_edges]
        valid_w = weights[valid_edges]
        unactivated_mask = ~activated[valid_u]
        valid_u = valid_u[unactivated_mask]
        valid_w = valid_w[unactivated_mask]
        valid_count = len(valid_u)
        if valid_count == 0:
            continue
        rand_values = rng.random(valid_count)
        activated_nodes = valid_u[rand_values < valid_w]
        unique_new = np.unique(activated_nodes)
        activated[unique_new] = True
        active[unique_new] = True
    rr = np.where(activated)[0]
    return rr


def optimized_node_selection(graph_shared, c, RR_sets, k, node_to_rr):
    communities_shm = shm.SharedMemory(name=graph_shared["communities_shm"])
    communities = np.ndarray((communities_shm.size // np.dtype(np.int32).itemsize,), dtype=np.int32, buffer=communities_shm.buf)
    n = communities.shape[0]
    coverage = c.copy()
    is_covered = np.zeros(len(RR_sets), dtype=bool)
    selected = []
    for _ in range(k):
        v = np.argmax(coverage)
        selected.append(v)
        for j in node_to_rr[v]:
            if is_covered[j]:
                continue
            is_covered[j] = True
            coverage[RR_sets[j]] -= 1
    return selected, np.sum(is_covered) * n / len(RR_sets)


def imm_cbga_sampling(graph_shared, k, epsilon, delta):
    communities_shm = shm.SharedMemory(name=graph_shared["communities_shm"])
    communities = np.ndarray((communities_shm.size // np.dtype(np.int32).itemsize,), dtype=np.int32, buffer=communities_shm.buf)
    n = communities.shape[0]
    theta = [0]
    epsilon2 = math.sqrt(2) * epsilon
    l = - math.log(delta) / math.log(n)
    c = np.zeros(n, dtype=np.int32)
    RR_set = list()
    obj_S = 0
    node_to_rr = defaultdict(list)
    for i in range(1, math.ceil(math.log2(n))):
        x = n / math.pow(2, i)
        theta_i = math.ceil((n * (2 + 2 / 3 * epsilon2) * (math.log(math.comb(n, k)) + l * math.log(n) + math.log(2) + math.log(math.log2(n)))) / (math.pow(epsilon2, 2) * x))
        theta.append(theta_i)
        RR_sets = parallel_generate_rr_sets(graph_shared, theta[i] - theta[i - 1])
        start_idx = len(RR_set)
        RR_set.extend(RR_sets)
        for q, rr in enumerate(RR_sets):
            j = start_idx + q
            c[rr] += 1
            for node in rr:
                node_to_rr[node].append(j)
        del RR_sets
        S, obj_S = optimized_node_selection(graph_shared, c, RR_set, k, node_to_rr)
        logging.info(f"Iteration {i} finished: number of rr-sets={theta_i}, x={x:.2f}, obj_S={obj_S:.2f}")
        if obj_S >= (1 + epsilon2) * x:
            logging.info(f"Stopping at iteration {i} because obj_S ({obj_S:.2f}) >= (1 + epsilon2) * x ({(1 + epsilon2) * x:.2f})")
            break
    LB = obj_S / (1 + epsilon2)
    del RR_set, RR_sets

    alpha = math.sqrt(l * math.log(n) + math.log(4))
    beta = math.sqrt((1 - 1 / math.e) * (math.log(math.comb(n, k)) + l * math.log(n) + math.log(4)))
    theta_0 = math.ceil((2 * n * math.pow((1 - 1 / math.e) * alpha + beta, 2)) / (LB * math.pow(epsilon, 2)))
    RR_set = parallel_generate_rr_sets(graph_shared, theta_0)
    c = np.zeros(n, dtype=np.int32)
    node_to_rr = defaultdict(list)
    for q, rr in enumerate(RR_set):
        c[rr] += 1
        for node in rr:
            node_to_rr[node].append(q)
    return RR_set, c, node_to_rr


def imm_optimized(graph_shared, graph_name, k, epsilon, delta, setR=10000):
    logging.info("Start IMM, graph:{}, P1={}, k={}, epsilon={}, delta={}".format(graph_name, P1, k, epsilon, delta))
    start_time = time.time()
    RR_set, c, node_to_rr = imm_cbga_sampling(graph_shared, k, epsilon, delta)
    sampling_time = time.time() - start_time
    S_return, obj_S = optimized_node_selection(graph_shared, c, RR_set, k, node_to_rr)
    running_time = time.time() - start_time
    logging.info("IMM Finished. Sampling Time: {:.2f}s, Running Time: {:.2f}s".format(sampling_time, running_time))
    logging.info("Selected {} Seed Nodes: {}".format(len(S_return), S_return[:min(len(S_return), 1000)]))
    logging.info("Estimated Objective: {:.2f}".format(obj_S))
    logging.info("Running Monte Carlo Simulation to validate seed set...")
    revenue_rho = estimate_revenue(graph_shared, seed_set=S_return, setR=setR)
    logging.info("Revenue rho by 10000 Monte Carlo simulations: {:.2f}".format(revenue_rho))

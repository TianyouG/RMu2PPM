import os

import numpy as np
import logging
import multiprocessing as mp
import multiprocessing.shared_memory as shm


P1 = 0.1
P2 = 1.0


def estimate_revenue(graph_shared, seed_set, setR=10000):
    logging.info("Start estimating revenue by MC...")
    num_workers = min(32, mp.cpu_count())
    with mp.Pool(num_workers) as pool:
        args = [(graph_shared, seed_set) for i in range(setR)]
        results = pool.starmap(_parallel_mc_worker, args, chunksize=max(1, setR // (num_workers * 2)))
    return np.mean(results)


def _parallel_mc_worker(graph_shared, seed_set):
    sources_shm = shm.SharedMemory(name=graph_shared["sources_shm"])
    targets_shm = shm.SharedMemory(name=graph_shared["targets_shm"])
    weights_shm = shm.SharedMemory(name=graph_shared["weights_shm"])
    communities_shm = shm.SharedMemory(name=graph_shared["communities_shm"])
    sources = np.ndarray((sources_shm.size // np.dtype(np.int32).itemsize,), dtype=np.int32, buffer=sources_shm.buf)
    targets = np.ndarray((targets_shm.size // np.dtype(np.int32).itemsize,), dtype=np.int32, buffer=targets_shm.buf)
    weights = np.ndarray((weights_shm.size // np.dtype(np.float32).itemsize,), dtype=np.float32, buffer=weights_shm.buf)
    communities = np.ndarray((communities_shm.size // np.dtype(np.int32).itemsize,), dtype=np.int32, buffer=communities_shm.buf)
    rng = np.random.default_rng(int.from_bytes(os.urandom(8), 'big'))
    return _vectorized_calculate_revenue(sources, targets, weights, communities, seed_set, rng)


def _vectorized_calculate_revenue(sources, targets, weights, communities, seed_set, rng):
    activated = np.zeros(len(communities), dtype=bool)
    activated[list(seed_set)] = True
    new_active = activated.copy()
    while np.any(new_active):
        active_nodes = np.where(new_active)[0]
        edge_mask = np.isin(sources, active_nodes)
        valid_targets = targets[edge_mask]
        valid_weights = weights[edge_mask]
        rand_values = rng.random(len(valid_targets))
        success = rand_values < valid_weights
        newly_activated = np.unique(valid_targets[success])
        new_active[:] = False
        new_active[newly_activated[~activated[newly_activated]]] = True
        activated |= new_active
    activated_indices = np.where(activated)[0]
    activated_communities = communities[activated_indices]
    unique_communities, community_sizes = np.unique(communities, return_counts=True)
    activated_counts = np.zeros_like(unique_communities)
    for i, comm_id in enumerate(unique_communities):
        activated_counts[i] = np.sum(activated_communities == comm_id)
    part1 = activated_counts * P1
    part2 = (community_sizes - activated_counts) * (activated_counts / community_sizes) * P2
    rho = np.sum(part1 + part2)
    return rho

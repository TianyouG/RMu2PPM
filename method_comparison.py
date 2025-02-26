import argparse
import os
import datetime
import logging
import ray

import multiprocessing.shared_memory as shm
import numpy as np

from utils.graph_loader import load_facebook, load_twitter, load_large_graph_optimized
from models.optimized_IMM_CBGA import imm_cbga_optimized
from models.optimized_IMM import imm_optimized
from models.optimized_IMM_weighted import imm_weighted_optimized
from models.optimized_IMM_modified import imm_modified_optimized
from models.monteCarloC import P1

parser = argparse.ArgumentParser()
parser.add_argument("--m", "--method", type=str, dest="method", choices=["opt_imm_cbga", "opt_imm", "opt_imm_weighted", "opt_imm_modified"], default='opt_imm_cbga', required=False)
parser.add_argument("--g", "--graph", type=str, dest="graph", choices=["facebook", "twitter", "email", "dblp", "youtube", "lj"], default="facebook", help='Which graph to use')
parser.add_argument("--k", type=int, default=50, help='size of seed set')
parser.add_argument("--e", "--epsilon", type=float, dest="epsilon", default=0.1, required=False, help='parameter of (epsilon-delta)-approximation')
parser.add_argument("--d", "--delta", type=float, dest="delta", default=None, required=False, help='parameter of (epsilon-delta)-approximation')
parser.add_argument("--R", "--setR", type=int, dest="setR", default=10000, help='the number of Monte Carlo simulations')
parser.add_argument("--num_re", "--num_repeat_experiment", type=int, dest="num_repeat_experiment", default=1, required=False, help="number of repeat experiments")
parser.add_argument("--sa", "--sa", type=str, dest="sa", default="n", choices=["n", "l", "u", "sa"], required=False, help="the parameter for SA strategy, only used when [m=='opt_imm_cbga']")
args = parser.parse_args()

log_dir = "log_{}".format(args.graph)
os.makedirs(log_dir, exist_ok=True)
ray.init(ignore_reinit_error=True)
log_filename = os.path.join(log_dir, "{}_k{}_P1_{}_{}.log".format(args.graph, args.k, str(int(P1 * 10)), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
logging.basicConfig(filename=log_filename, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Program started.")

if __name__ == '__main__':
    if args.graph == "twitter":
        graph_ref = load_twitter(file_path="data/twitter_combined.txt", community_path="data/twitter_community.txt", saved_hdf5_path="data/labeled_twitter.hdf5")
    elif args.graph == "facebook":
        graph_ref = load_facebook(file_path="data/facebook_combined.txt", community_path="data/facebook_community.txt", saved_hdf5_path="data/labeled_facebook.hdf5")
    else:
        graph_ref = load_large_graph_optimized(file_path="data/com-{}.preprocessed.ungraph.txt".format(args.graph), community_path="data/com-{}.preprocessed.cmty.txt".format(args.graph), saved_hdf5_path="data/labeled_{}.hdf5".format(args.graph), sharpen_boundary=True)
    graph = ray.get(graph_ref)
    num_edges = graph.ecount()
    num_nodes = graph.vcount()
    args.delta = 1 / num_nodes
    CBGA = (getattr(args, "method", None) == "opt_imm_cbga")
    sources_shm = shm.SharedMemory(create=True, size=num_edges * np.dtype(np.int32).itemsize)
    targets_shm = shm.SharedMemory(create=True, size=num_edges * np.dtype(np.int32).itemsize)
    weights_shm = shm.SharedMemory(create=True, size=num_edges * np.dtype(np.float32).itemsize)
    communities_shm = shm.SharedMemory(create=True, size=num_nodes * np.dtype(np.int32).itemsize)
    np.ndarray((num_edges,), dtype=np.int32, buffer=sources_shm.buf)[:] = [e.source for e in graph.es]
    np.ndarray((num_edges,), dtype=np.int32, buffer=targets_shm.buf)[:] = [e.target for e in graph.es]
    np.ndarray((num_edges,), dtype=np.float32, buffer=weights_shm.buf)[:] = graph.es["weight"]
    np.ndarray((num_nodes,), dtype=np.int32, buffer=communities_shm.buf)[:] = graph.vs["community"]
    if CBGA:
        sources_valid_shm = shm.SharedMemory(create=True, size=(num_edges - graph["numBoundaryEdge"]) * np.dtype(np.int32).itemsize)
        targets_valid_shm = shm.SharedMemory(create=True, size=(num_edges - graph["numBoundaryEdge"]) * np.dtype(np.int32).itemsize)
        weights_valid_shm = shm.SharedMemory(create=True, size=(num_edges - graph["numBoundaryEdge"]) * np.dtype(np.float32).itemsize)
        np.ndarray((num_edges - graph["numBoundaryEdge"],), dtype=np.int32, buffer=sources_valid_shm.buf)[:] = [e.source for idx, e in enumerate(graph.es) if not graph.es[idx]["isBoundary"]]
        np.ndarray((num_edges - graph["numBoundaryEdge"],), dtype=np.int32, buffer=targets_valid_shm.buf)[:] = [e.target for idx, e in enumerate(graph.es) if not graph.es[idx]["isBoundary"]]
        np.ndarray((num_edges - graph["numBoundaryEdge"],), dtype=np.float32, buffer=weights_valid_shm.buf)[:] = [e["weight"] for idx, e in enumerate(graph.es) if not graph.es[idx]["isBoundary"]]
    del graph, graph_ref
    ray.shutdown()
    if CBGA:
        graph_shared = {
            "sources_shm": sources_shm.name,
            "targets_shm": targets_shm.name,
            "weights_shm": weights_shm.name,
            "communities_shm": communities_shm.name,
            "sources_valid_shm": sources_valid_shm.name,
            "targets_valid_shm": targets_valid_shm.name,
            "weights_valid_shm": weights_valid_shm.name
        }
    else:
        graph_shared = {
            "sources_shm": sources_shm.name,
            "targets_shm": targets_shm.name,
            "weights_shm": weights_shm.name,
            "communities_shm": communities_shm.name
        }
    for _ in range(args.num_repeat_experiment):
        if args.method == "opt_imm_cbga":
            imm_cbga_optimized(graph_shared, graph_name="args.graph", k=args.k, epsilon=args.epsilon, delta=args.delta, setR=args.setR, sa=args.sa)
        elif args.method == "opt_imm":
            imm_optimized(graph_shared, graph_name="args.graph", k=args.k, epsilon=args.epsilon, delta=args.delta, setR=args.setR)
        elif args.method == "opt_imm_weighted":
            imm_weighted_optimized(graph_shared, graph_name="args.graph", k=args.k, epsilon=args.epsilon, delta=args.delta, setR=args.setR)
        elif args.method == "opt_imm_modified":
            imm_modified_optimized(graph_shared, graph_name="args.graph", k=args.k, epsilon=args.epsilon, delta=args.delta, setR=args.setR)

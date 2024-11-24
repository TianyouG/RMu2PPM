import argparse
from utils.graph_generator import generate_lfr_graph
from utils.graph_loader import load_facebook, load_twitter, load_email_Eu
from models.general_greedy import greedy
from models.CBGA import community_based_greedy
from models.sandwich_approximation import sandwich_approximation
from models.baseline import rand, high_degree
from models.CELF_greedy import greedy_celf_plus
from models.CELF_CBGA import cbga_celf_plus
from utils.boundary_revenue_estimator import boundary_revenue_estimator

parser = argparse.ArgumentParser()
parser.add_argument("--m", "--method", type=str, dest="method",
                    choices=["rand", "in_degree", "out_degree", "greedy", "cbga", "sa_greedy", "sa_cbga", "greedy_celf",
                             "cbga_celf", "sa_greedy_celf", "sa_cbga_celf", "boundary_revenue_estimate"], default='greedy')
parser.add_argument("--g", "--graph", type=str, dest="graph", choices=["synthetic", "facebook", "twitter", "email-Eu"], default="synthetic", help='Which graph to use')
parser.add_argument("--N", type=int, default=1000, required=False, help='Size of the synthetic graph')
parser.add_argument("--tau1", type=float, default=3.0, required=False, help='parameter of LFR_benchmark: degree distribution exponent')
parser.add_argument("--tau2", type=float, default=1.5, required=False, help='parameter of LFR_benchmark: community size distribution exponent')
parser.add_argument("--mu", type=float, default=0.025, required=False, help='parameter of LFR_benchmark: mixing coefficient')
parser.add_argument("--min_degree", type=int, default=2, required=False, help='parameter of LFR_benchmark: minimum degree')
parser.add_argument("--max_degree", type=int, default=50, required=False, help='parameter of LFR_benchmark: maximum degree')
parser.add_argument("--min_community", type=int, default=50, required=False, help='parameter of LFR_benchmark: minimum size of community')
parser.add_argument("--max_community", type=int, default=300, required=False, help='parameter of LFR_benchmark: maximum size of the community')
parser.add_argument("--s", "--seed", type=int, dest="seed", default=4561569, required=False, help='random seed to generate graph by LFR benchmark')
parser.add_argument("--k", type=int, default=20, help='size of seed set')
parser.add_argument("--e", "--epsilon", type=float, dest="epsilon", default=0.2, required=False, help='parameter of (epsilon-delta)-approximation')
parser.add_argument("--d", "--delta", type=float, dest="delta", default=0.1, required=False, help='parameter of (epsilon-delta)-approximation')
parser.add_argument("--R", "--setR", type=int, dest="setR", default=None, help='the number of Monte Carlo simulations')
parser.add_argument("--rt", "--repeat_times", type=int, dest="repeat_times", default=100, required=False, help='only needed if method="rand", the number of times to repeat function rand')

args = parser.parse_args()

if __name__ == '__main__':
    if args.graph == "synthetic":
        graph = generate_lfr_graph(args.N, tau1=args.tau1, tau2=args.tau2, mu=args.mu, min_degree=args.min_degree,
                                   max_degree=args.max_degree, min_community=args.min_community,
                                   max_community=args.max_community, seed=args.seed)
    elif args.graph == "facebook":
        graph = load_facebook(file="data/facebook_combined.txt", community_file="data/facebook_community.txt")
    elif args.graph == "twitter":
        graph = load_twitter(file="data/twitter_combined.txt", community_file="data/twitter_community.txt")
    elif args.graph == "email-Eu":
        graph = load_email_Eu(file="data/email-Eu-core.txt", community_file="data/email-Eu-core-department-labels.txt")
    else:
        raise NameError("Unknown graph (--g): {}".format(args.graph))
    communities = graph["communities"]

    if args.method == "greedy":
        seed_set, rtime = greedy(graph, communities, k=args.k, epsilon=args.epsilon, delta=args.delta, setR=args.setR)
    elif args.method == "cbga":
        seed_set, rtime = community_based_greedy(graph, communities, k=args.k, epsilon=args.epsilon, delta=args.delta, setR=args.setR)
    elif args.method == "sa_greedy":
        seed_set, rtime = sandwich_approximation(graph, communities, k=args.k, epsilon=args.epsilon, delta=args.delta, isCBGA=False, setR=args.setR)
    elif args.method == "sa_cbga":
        seed_set, rtime = sandwich_approximation(graph, communities, k=args.k, epsilon=args.epsilon, delta=args.delta, isCBGA=True, setR=args.setR)
    elif args.method == "rand":
        seed_set, rtime = rand(graph, communities, k=args.k, epsilon=args.epsilon, delta=args.delta, setR=args.setR, repeat_times=args.repeat_times)
    elif args.method == "in_degree":
        seed_set, rtime = high_degree(graph, communities, k=args.k, epsilon=args.epsilon, delta=args.delta, setR=args.setR, mode="in")
    elif args.method == "out_degree":
        seed_set, rtime = high_degree(graph, communities, k=args.k, epsilon=args.epsilon, delta=args.delta, setR=args.setR, mode="out")
    elif args.method == "greedy_celf":
        seed_set, rtime = greedy_celf_plus(graph, communities, k=args.k, epsilon=args.epsilon, delta=args.delta, setR=args.setR)
    elif args.method == "cbga_celf":
        seed_set, rtime = cbga_celf_plus(graph, communities, k=args.k, epsilon=args.epsilon, delta=args.delta, setR=args.setR)
    elif args.method == "sa_greedy_celf":
        seed_set, rtime = sandwich_approximation(graph, communities, k=args.k, epsilon=args.epsilon, delta=args.delta, isCBGA=False, celf=True, setR=args.setR)
    elif args.method == "sa_cbga_celf":
        seed_set, rtime = sandwich_approximation(graph, communities, k=args.k, epsilon=args.epsilon, delta=args.delta, isCBGA=True, celf=True, setR=args.setR)
    elif args.method == "boundary_revenue_estimate":
        res = boundary_revenue_estimator(graph, communities)

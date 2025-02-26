# RMu2PPM
 Code for Paper: Revenue Maximization for Products with Presale Phase

Usage
example: python method_comparision.py --m opt_imm --g facebook --k 50 --e 0.5 --R 10000

Options:

--m: which method to use
choices=["opt_imm_cbga", "opt_imm", "opt_imm_weighted", "opt_imm_modified"]
default="opt_imm_cbga"

--g: which graph to use
choices=["facebook", "twitter", "email", "dblp", "youtube", "lj"]
default="facebook"

--k: seed size
default=50

--e: parameter \epsilon
default=0.1

--d: parameter \delta
default=1/N

--R: The number of MC simulations
default=10000

--num_re: Number of repeat experiments
default=1

--sa: The parameter for SA strategy, only used when [m=='opt_imm_cbga']
choices=["n", "l", "u", "sa"]
default="n"

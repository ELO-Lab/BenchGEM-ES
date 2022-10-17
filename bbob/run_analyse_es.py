import os
import numpy as np
from strategies import *
from benchmarks import *

def main(args):
    lrs    = [0.5, 0.05, 0.005, 0.1, 0.01, 0.001]
    seeds  = list(range(2016, 2025 + 1))
    solver = ES(args.sigma, 0, args.pop_size, True)

    for seed in seeds:
        data_table = {}
        for lr in lrs:
            solver.learning_rate = lr
            p_manager  = Manager(args.dimension, seed)
            problem    = p_manager.get_function(args.func_id)
            propose_x0 = p_manager.get_initial_solution()
            # Not use early-stopping
            solver.optimize(np.copy(propose_x0), problem, args.max_iters, args.max_evals, False, args.threads)
            data_table[lr] = [np.array(solver.history_loss)]
        
        # Save to .npy
        file_name = "ES_%s_%s_%s.npy" % (str(seed), str(args.func_id), str(args.dimension))
        try:
            try:
                os.mkdir(args.save_dir)
            except FileExistsError:
                pass
            np.save(args.save_dir + file_name, data_table)
        except IOError:
            print("I/O error")

if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--max_iters', type=int, default=int(100))
    parse.add_argument('--max_evals', type=int, default=int(10000))
    parse.add_argument('--dimension', type=int, default=int(10))
    parse.add_argument('--func_id', type=int, default=1)
    parse.add_argument('--sigma', type=float, default=0.01)
    parse.add_argument('--pop_size', type=int, default=100)   
    parse.add_argument('--save_dir', type=str, default='./logs/')
    parse.add_argument('--threads', type=int, default=1)

    args = parse.parse_args()
    main(args)

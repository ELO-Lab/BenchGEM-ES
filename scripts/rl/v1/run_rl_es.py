import argparse, os, gym, ray
import numpy as np
from strategies_v1 import ES
from workers import Worker
from policies import LinearPolicy

class RLController(object):
    def __init__(self):
        self.seed 	= args.seed
        self.env_name   = args.env_name
        self.env 	= gym.make(self.env_name)
        self.state_dim  = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # Init parameters
        self._init_parameters()

        # Initialize workers with different random seeds
        self.workers = [Worker.remote(self.seed + 7 * i, self.env_name, self.policy_params, self.max_ep_len) 
                        for i in range(self.num_workers)]
        
        # Initialize policy
        self.policy = LinearPolicy(self.policy_params)
        self.params = self.policy.get_params()

        # Initialize solver
        self.solver = ES(self.sigma, self.learning_rate, self.pop_size, self.elite_size, self.seed)

        print("Initialization of ES complete.")
        
    def _init_parameters(self):
        self.max_ep_len  = args.max_ep_len
        self.num_workers = args.num_workers
        self.max_iters   = args.max_iters
        self.save_dir 	 = args.save_dir

        self.sigma 	   = args.sigma
        self.learning_rate = args.learning_rate
        self.pop_size 	   = args.pop_size
        self.elite_size    = args.elite_size

        self.policy_params = {'filter'    : args.filter, 
                              'state_dim' : self.state_dim, 
                              'action_dim': self.action_dim}
    
    def aggregate_rollouts(self, set_params, eval=False):
        rollout_ids, worker_id = [], 0
        for param in set_params:
            param_id     = ray.put(param)
            rollout_ids += [self.workers[worker_id].do_rollout.remote(param_id, eval=eval)]
            worker_id    = (worker_id + 1) % self.num_workers
        
        results = ray.get(rollout_ids)
        # We need minimize objective function
        all_rollout_rewards = [-result["rollout_reward"] for result in results]

        return all_rollout_rewards

    def train(self):
        losses = []
        self.solver.initialize(np.zeros_like(self.params))
        for iter in range(self.max_iters):
            set_params 		= self.solver.ask()
            all_rollout_rewards = self.aggregate_rollouts(set_params, eval=False)
            self.solver.tell(all_rollout_rewards)

            # Evaluate 20 times
            losses.append(np.mean(self.aggregate_rollouts([self.solver.mu] * 20, eval=True)))
            if (iter + 1) % 10 == 0:
                print("Iters: %d, Average Loss: %f" % (iter + 1, losses[-1]))
        
            # Get statistics from all workers
            for i in range(self.num_workers):
                self.policy.state_filter.update(ray.get(self.workers[i].get_filter.remote()))
            self.policy.state_filter.stats_increment()

            # Make sure master filter buffer is clear
            self.policy.state_filter.clear_buffer()
            # Sync all workers
            filter_id = ray.put(self.policy.state_filter)
            setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
            # Waiting for sync of all workers
            ray.get(setting_filters_ids)

            increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
            # Waiting for increment of all workers
            ray.get(increment_filters_ids)
        
        # Save to .npy
        file_name = "ES_%s_%s.npy" % (str(self.env_name), str(self.seed))
        try:
            try:
                os.mkdir(self.save_dir)
            except FileExistsError:
                pass
            np.save(self.save_dir + file_name, losses)
        except IOError:
            print("I/O error")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', type=int, default=2016)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=int(1e2))
    parser.add_argument('--filter', type=str, default='MeanStdFilter')
    parser.add_argument('--save_dir', type=str, default='./logs_rl/')
    parser.add_argument('--sigma', type=float, default=0.05)
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--pop_size', type=int, default=32)
    parser.add_argument('--elite_size', type=int, default=32)
    args = parser.parse_args()

    ray.init()
    controller = RLController()
    controller.train()

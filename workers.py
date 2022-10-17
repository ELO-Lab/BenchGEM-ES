import gym, ray
from policies import LinearPolicy

@ray.remote
class Worker(object):
    def __init__(self, env_seed,
                 env_name      = 'HalfCheetah-v2',
                 policy_params = None,
                 max_ep_len    = 1000):

        # Initialize OpenAI environment for each worker
        self.env = gym.make(env_name)
        self.env.seed(env_seed)

        self.policy     = LinearPolicy(policy_params)
        self.max_ep_len = max_ep_len
        
    def rollout(self):
        observation, total_reward = self.env.reset(), 0.
        for i in range(self.max_ep_len):
            action = self.policy.act(observation)
            observation, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done:
                break
        
        return total_reward
    
    def do_rollout(self, params, eval=False):
        if eval:
            self.policy.update_filter = False
        else:
            self.policy.update_filter = True

        self.policy.update_params(params)
        rollout_reward = self.rollout()

        return {"rollout_reward": rollout_reward}
    
    def stats_increment(self):
        self.policy.state_filter.stats_increment()

    def get_filter(self):
        return self.policy.state_filter

    def sync_filter(self, other):
        self.policy.state_filter.sync(other)








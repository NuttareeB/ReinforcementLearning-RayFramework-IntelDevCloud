import gym
import torch
import time
import os
import ray
import numpy as np

from tqdm import tqdm
from random import uniform, randint

import io
import base64
from IPython.display import HTML

from dqn_model import DQNModel
from memory import ReplayBuffer

import matplotlib.pyplot as plt

from memory_remote import ReplayBuffer_remote
from dqn_model import _DQNModel
import torch
from custom_cartpole import CartPoleEnv

FloatTensor = torch.FloatTensor

# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole_distributed'

# Move left, Move right
ACTION_DICT = {
    "LEFT": 0,
    "RIGHT":1
}

# Set result saveing floder
result_folder = ENV_NAME
result_file = ENV_NAME + "/results_16_cw_16_ew-5.txt"
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)
torch.set_num_threads(12)

def plot_result(total_rewards ,learning_num, legend):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)
        
    plt.figure(num = 1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.savefig("Distributed_DQN_16_cw_16_ew-5.png")
    plt.show()
    
ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)

@ray.remote
class model_server():
    def __init__(self, env, hyper_params, memory, test_interval, action_space):
        state = env.reset()
        input_len = len(state)
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate = hyper_params['learning_rate'])
        self.use_target_model = hyper_params['use_target_model']
        if self.use_target_model:
            self.target_model = DQNModel(input_len, output_len)
        self.memory = memory
        
        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.beta = hyper_params['beta']
        
        self.test_interval = test_interval
        
        self.collector_done = False
        self.evaluator_done = False
        self.episode = 0
        self.steps = 0
        self.best_reward = 0
        self.learning = True
        self.action_space = action_space
        
        self.prev_q_net = []
        self.result_count = 0
        self.results = []
        self.learning_episodes = training_episodes
        
    def get_steps(self):
        return self.steps
    
    def get_learn_status(self):
        if self.episode >= self.learning_episodes:
            self.collector_done = True
            
        return self.collector_done
    
    def predict_eval_model(self, state):
        return self.eval_model.predict(state)
    
    def replace_target_model(self):
        self.target_model.replace(self.eval_model)
        
    def update_batch(self):
        
        self.steps += self.update_steps;
        
        if ray.get(self.memory.__len__.remote()) < self.batch_size or self.collector_done:
            return

        batch = ray.get(self.memory.sample.remote(self.batch_size))

        (states, actions, reward, next_states, is_terminal) = batch
        
        states = states
        next_states = next_states
        terminal = FloatTensor([1 if t else 0 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size,
                                   dtype=torch.long)
        
        # Current Q Values
        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]
        
        # Calculate target
        if self.use_target_model:
            actions, q_next = self.target_model.predict_batch(next_states)
        else:
            actions, q_next = self.eval_model.predict_batch(next_states)
            
        q_targets = []
        
        for i, is_terminal in enumerate(terminal):
            q_tar = reward[i] if is_terminal == 1 else reward[i] + (self.beta * torch.max(q_next, 1).values[i].data)
            q_targets.append(q_tar)
            
        q_target = FloatTensor(q_targets)
        
        # update model
        self.eval_model.fit(q_values, q_target)
        
        if self.episode // self.test_interval + 1 > len(self.prev_q_net):
            eval_model_id = ray.put(self.eval_model)
            self.prev_q_net.append(eval_model_id)
            
        return self.steps
    
    # evalutor
    def add_result(self, result, num):
        self.results.append(result)
    
    def get_results(self):
        return self.results
    
    def ask_evaluation(self):
        if len(self.prev_q_net) > self.result_count:
            num = self.result_count
            eval_q_net = self.prev_q_net[num]
            self.result_count += 1
            self.episode += self.test_interval
            return eval_q_net, False, num
        else:
            if self.episode >= self.learning_episodes:
                self.evaluator_done = True
            return [], self.evaluator_done, None

@ray.remote    
def collecting_worker(model_server, env, memory, initial_epsilon, final_epsilon, epsilon_decay_steps, test_interval, model_replace_freq, update_steps, use_target_model, action_space):
    def greedy_policy(curr_state):
        return ray.get(model_server.predict_eval_model.remote(curr_state))

    def linear_decrease(initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate
    
    def explore_or_exploit_policy(curr_state):
        p = uniform(0, 1)
        
        steps = ray.get(model_server.get_steps.remote())
        epsilon = linear_decrease(initial_epsilon, 
                               final_epsilon,
                               steps,
                               epsilon_decay_steps)
        
        if p < epsilon:
            return randint(0, action_space - 1)
        else:
            return greedy_policy(curr_state)
        
    
    while True:
        learn_done = ray.get(model_server.get_learn_status.remote())
        if learn_done:
            break
            
        for episode in tqdm(range(test_interval), desc="Training"):
            state = env.reset()
            done = False
            steps = 0

            while steps < env._max_episode_steps and not done:
                steps += 1
                action = explore_or_exploit_policy(state)
                next_state, reward, done, _ = env.step(action)
                
                # add experience from explore-exploit policy to memory
                memory.add.remote(state, action, reward, next_state, done)
                
                # update the model every 'update_steps' of experience
                if (steps % update_steps) == 0:
                    model_server.update_batch.remote()
                    
                server_steps = ray.get(model_server.get_steps.remote())
                
                # update the target network (if the target network is being used) every 'model_replace_freq' of experiences 
                if use_target_model and (server_steps % model_replace_freq) == 0:
                    model_server.replace_target_model.remote()
                
                state = next_state
        
@ray.remote
def evaluation_worker(model_server, env, trials = 30):
    def greedy_policy(curr_state, eval_q_net):
        return eval_q_net.predict(curr_state)
    
    while True:
        eval_model_id, done, num = ray.get(model_server.ask_evaluation.remote())
        eval_q_net = ray.get(eval_model_id)

        if done:
            break
        if eval_q_net == []:
            continue
        total_reward = 0
        for _ in tqdm(range(trials), desc="Evaluating"):
            state = env.reset()
            done = False
            steps = 0

            while steps < env._max_episode_steps and not done:
                steps += 1
                action = greedy_policy(state, eval_q_net)
                state, reward, done, _ = env.step(action)
                total_reward += reward

        avg_reward = total_reward / trials

        model_server.add_result.remote(avg_reward, num)
        
        print(avg_reward)
        f = open(result_file, "a+")
        f.write(str(avg_reward) + "\n")
        f.close()
#     return avg_reward

class distributed_DQN_agent(object):
    def __init__(self, env, hyper_params, action_space = len(ACTION_DICT), training_episodes = 10000, test_interval = 50, cw_num = 4, ew_num = 4):
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        
        self.beta = hyper_params['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']

        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']
        self.use_target_model = hyper_params['use_target_model']
        
        self.memory = ReplayBuffer_remote.remote(hyper_params['memory_size'])
        
        self.action_space = action_space
        self.training_episodes = training_episodes
        self.test_interval = test_interval
        self.cw_num = cw_num
        self.ew_num = ew_num
        
        self.model_server = model_server.remote(env, hyper_params, self.memory, test_interval, action_space)
        
    def learn_and_evaluate(self):
        workers_id = []
        
        for i in range(self.cw_num):
            worker_cw = collecting_worker.remote(self.model_server, 
                                                 self.env, 
                                                 self.memory, 
                                                 self.initial_epsilon, 
                                                 self.final_epsilon,
                                                 self.epsilon_decay_steps, 
                                                 self.test_interval, 
                                                 self.model_replace_freq, 
                                                 self.update_steps, 
                                                 self.use_target_model,
                                                 self.action_space)
            workers_id.append(worker_cw)

        for i in range(self.ew_num):
            worker_ew = evaluation_worker.remote(self.model_server, self.env)
            workers_id.append(worker_ew)
        
        ray.wait(workers_id, len(workers_id))
        return ray.get(self.model_server.get_results.remote())
    
hyperparams_CartPole = {
    'epsilon_decay_steps' : 100000, 
    'final_epsilon' : 0.1,
    'batch_size' : 32, 
    'update_steps' : 10, 
    'memory_size' : 2000, 
    'beta' : 0.99, 
    'model_replace_freq' : 2000,
    'learning_rate' : 0.0003,
    'use_target_model': True
}

start_time = time.time()
training_episodes, test_interval = 10000, 50
distributed_dqn_agent = distributed_DQN_agent(CartPoleEnv(), 
                                              hyperparams_CartPole, 
                                              action_space = len(ACTION_DICT), 
                                              training_episodes = training_episodes, 
                                              test_interval = test_interval, 
                                              cw_num = 16, 
                                              ew_num = 16)
result = distributed_dqn_agent.learn_and_evaluate()
run_time = time.time() - start_time
print("Learning time:\n", run_time)

plot_result(result, test_interval, ["batch_update with target_model: Distributed"])
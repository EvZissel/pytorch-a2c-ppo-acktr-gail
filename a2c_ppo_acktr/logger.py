import numpy as np
import pandas as pd
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import time

class Logger(object):
    
    def __init__(self, n_envs):
        self.start_time = time.time()
        self.n_envs = n_envs

        self.episode_rewards = []
        self.episode_rewards_test = []
        self.episode_rewards_train = []
        for _ in range(n_envs):
            self.episode_rewards.append([])
            self.episode_rewards_train.append([])
            self.episode_rewards_test.append([])

        self.episode_len_buffer = deque(maxlen = n_envs)
        self.episode_len_buffer_train = deque(maxlen=n_envs)
        self.episode_len_buffer_test = deque(maxlen=n_envs)
        self.episode_reward_buffer = deque(maxlen = n_envs)
        self.episode_reward_buffer_train = deque(maxlen=n_envs)
        self.episode_reward_buffer_test = deque(maxlen=n_envs)


        self.num_episodes = 0
        self.num_episodes_train = 0
        self.num_episodes_test = 0

    def feed_eval(self, rew_batch_train, done_batch_train, rew_batch_test, done_batch_test):
        steps = rew_batch_train.shape[0]
        rew_batch_train = rew_batch_train.T
        done_batch_train = done_batch_train.T
        rew_batch_test = rew_batch_test.T
        done_batch_test = done_batch_test.T

        for i in range(self.n_envs):
            for j in range(steps):
                self.episode_rewards_test[i].append(rew_batch_test[i][j])
                self.episode_rewards_train[i].append(rew_batch_train[i][j])
                if done_batch_train[i][j]:
                    self.episode_len_buffer_train.append(len(self.episode_rewards_train[i]))
                    self.episode_reward_buffer_train.append(np.sum(self.episode_rewards_train[i]))
                    self.episode_rewards_train[i] = []
                    self.num_episodes_train += 1
                if done_batch_test[i][j]:
                    self.episode_len_buffer_test.append(len(self.episode_rewards_test[i]))
                    self.episode_reward_buffer_test.append(np.sum(self.episode_rewards_test[i]))
                    self.episode_rewards_test[i] = []
                    self.num_episodes_test += 1

    def feed_train(self, rew_batch, done_batch):
        steps = rew_batch.shape[0]
        rew_batch = rew_batch.T
        done_batch = done_batch.T

        for i in range(self.n_envs):
            for j in range(steps):
                self.episode_rewards[i].append(rew_batch[i][j])
                if done_batch[i][j]:
                    self.episode_len_buffer.append(len(self.episode_rewards[i]))
                    self.episode_reward_buffer.append(np.sum(self.episode_rewards[i]))
                    self.episode_rewards[i] = []
                    self.num_episodes += 1


    def get_episode_statistics(self):
        episode_statistics = {}
        episode_statistics['Rewards/max_episodes']  = {'train': np.max(self.episode_reward_buffer),
                                                       'train_val': np.max(self.episode_reward_buffer_train),
                                                       'test':np.max(self.episode_reward_buffer_test)}
        episode_statistics['Rewards/mean_episodes'] = {'train': np.mean(self.episode_reward_buffer),
                                                       'train_val': np.mean(self.episode_reward_buffer_train),
                                                       'test': np.mean(self.episode_reward_buffer_test)}
        episode_statistics['Rewards/min_episodes']  = {'train': np.min(self.episode_reward_buffer),
                                                       'train_val': np.min(self.episode_reward_buffer_train),
                                                       'test': np.min(self.episode_reward_buffer_test)}

        episode_statistics['Len/max_episodes']  = {'train': np.max(self.episode_len_buffer),
                                                   'train_val': np.max(self.episode_len_buffer_train),
                                                   'test': np.max(self.episode_len_buffer_test)}
        episode_statistics['Len/mean_episodes'] = {'train': np.mean(self.episode_len_buffer),
                                                   'train_val': np.mean(self.episode_len_buffer_train),
                                                   'test': np.mean(self.episode_len_buffer_test)}
        episode_statistics['Len/min_episodes']  = {'train': np.min(self.episode_len_buffer),
                                                   'train_val': np.min(self.episode_len_buffer_train),
                                                   'test': np.min(self.episode_len_buffer_test)}
        return episode_statistics

    def get_train_statistics(self):
        train_statistics = {}
        train_statistics['Rewards_max_episodes'] = np.max(self.episode_reward_buffer)

        train_statistics['Rewards_mean_episodes'] = np.mean(self.episode_reward_buffer)

        train_statistics['Rewards_median_episodes'] = np.median(self.episode_reward_buffer)

        train_statistics['Rewards_min_episodes'] = np.min(self.episode_reward_buffer)

        train_statistics['Len_max_episodes'] = np.max(self.episode_len_buffer)

        train_statistics['Len_mean_episodes'] = np.mean(self.episode_len_buffer)

        train_statistics['Len_min_episodes'] = np.min(self.episode_len_buffer)

        return train_statistics
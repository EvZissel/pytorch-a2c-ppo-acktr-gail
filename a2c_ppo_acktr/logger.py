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
        self.episode_rewards_val = []
        self.episode_rewards_test = []
        self.episode_rewards_train = []
        self.episode_rewards_test_oracle = []
        self.episode_rewards_train_oracle = []
        self.episode_rewards_train_val = []
        self.episode_rewards_test_nondet = []
        for _ in range(n_envs):
            self.episode_rewards.append([])
            self.episode_rewards_val.append([])
            self.episode_rewards_train.append([])
            self.episode_rewards_train_oracle.append([])
            self.episode_rewards_train_val.append([])
            self.episode_rewards_test.append([])
            self.episode_rewards_test_oracle.append([])
            self.episode_rewards_test_nondet.append([])

        self.episode_len_buffer = deque(maxlen = n_envs)
        self.episode_len_buffer_val = deque(maxlen = n_envs)
        self.episode_len_buffer_train = deque(maxlen=n_envs)
        self.episode_len_buffer_train_oracle = deque(maxlen=n_envs)
        self.episode_len_buffer_train_val = deque(maxlen=n_envs)
        self.episode_len_buffer_test = deque(maxlen=n_envs)
        self.episode_len_buffer_test_oracle = deque(maxlen=n_envs)
        self.episode_len_buffer_test_nondet = deque(maxlen=n_envs)
        self.episode_reward_buffer = deque(maxlen = n_envs)
        self.episode_reward_buffer_val = deque(maxlen = n_envs)
        self.episode_reward_buffer_train = deque(maxlen=n_envs)
        self.episode_reward_buffer_train_oracle = deque(maxlen=n_envs)
        self.episode_reward_buffer_train_val = deque(maxlen=n_envs)
        self.episode_reward_buffer_test = deque(maxlen=n_envs)
        self.episode_reward_buffer_test_oracle = deque(maxlen=n_envs)
        self.episode_reward_buffer_test_nondet = deque(maxlen=n_envs)


        self.num_episodes = 0
        self.num_episodes_val = 0
        self.num_episodes_train = 0
        self.num_episodes_train_oracle = 0
        self.num_episodes_train_val = 0
        self.num_episodes_test = 0
        self.num_episodes_test_oracle = 0
        self.num_episodes_test_nondet = 0

    def feed_eval(self, rew_batch_train, done_batch_train, rew_batch_test, done_batch_test, rew_batch_train_oracle, done_batch_train_oracle, rew_batch_test_oracle, done_batch_test_oracle,
                  rew_batch_train_val, done_batch_train_val, rew_batch_test_nondet, done_batch_test_nondet):

        steps = rew_batch_train.shape[0]
        rew_batch_train = rew_batch_train.T
        done_batch_train = done_batch_train.T
        rew_batch_test = rew_batch_test.T
        done_batch_test = done_batch_test.T
        rew_batch_train_oracle = rew_batch_train_oracle.T
        done_batch_train_oracle = done_batch_train_oracle.T
        rew_batch_test_oracle = rew_batch_test_oracle.T
        done_batch_test_oracle = done_batch_test_oracle.T
        rew_batch_train_val = rew_batch_train_val.T
        done_batch_train_val = done_batch_train_val.T
        rew_batch_test_nondet = rew_batch_test_nondet.T
        done_batch_test_nondet = done_batch_test_nondet.T
        for i in range(self.n_envs):
            for j in range(steps):
                self.episode_rewards_test[i].append(rew_batch_test[i][j])
                self.episode_rewards_train[i].append(rew_batch_train[i][j])
                self.episode_rewards_test_oracle[i].append(rew_batch_test_oracle[i][j])
                self.episode_rewards_train_oracle[i].append(rew_batch_train_oracle[i][j])
                self.episode_rewards_train_val[i].append(rew_batch_train_val[i][j])
                self.episode_rewards_test_nondet[i].append(rew_batch_test_nondet[i][j])
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
                if done_batch_train_oracle[i][j]:
                    self.episode_len_buffer_train_oracle.append(len(self.episode_rewards_train[i]))
                    self.episode_reward_buffer_train_oracle.append(np.sum(self.episode_rewards_train[i]))
                    self.episode_rewards_train_oracle[i] = []
                    self.num_episodes_train_oracle += 1
                if done_batch_test_oracle[i][j]:
                    self.episode_len_buffer_test_oracle.append(len(self.episode_rewards_test[i]))
                    self.episode_reward_buffer_test_oracle.append(np.sum(self.episode_rewards_test[i]))
                    self.episode_rewards_test_oracle[i] = []
                    self.num_episodes_test_oracle += 1
                if done_batch_train_val[i][j]:
                    self.episode_len_buffer_train_val.append(len(self.episode_rewards_train_val[i]))
                    self.episode_reward_buffer_train_val.append(np.sum(self.episode_rewards_train_val[i]))
                    self.episode_rewards_train_val[i] = []
                    self.num_episodes_train_val += 1
                if done_batch_test_nondet[i][j]:
                    self.episode_len_buffer_test_nondet.append(len(self.episode_rewards_test_nondet[i]))
                    self.episode_reward_buffer_test_nondet.append(np.sum(self.episode_rewards_test_nondet[i]))
                    self.episode_rewards_test_nondet[i] = []
                    self.num_episodes_test_nondet += 1

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


    def feed_val(self, rew_batch, done_batch):
        steps = rew_batch.shape[0]
        rew_batch = rew_batch.T
        done_batch = done_batch.T

        for i in range(self.n_envs):
            for j in range(steps):
                self.episode_rewards_val[i].append(rew_batch[i][j])
                if done_batch[i][j]:
                    self.episode_len_buffer_val.append(len(self.episode_rewards_val[i]))
                    self.episode_reward_buffer_val.append(np.sum(self.episode_rewards_val[i]))
                    self.episode_rewards_val[i] = []
                    self.num_episodes_val += 1


    def get_episode_statistics(self):
        episode_statistics = {}
        episode_statistics['Rewards/max_episodes']  = {'train': np.max(self.episode_reward_buffer),
                                                       'train_eval': np.max(self.episode_reward_buffer_train),
                                                       'train_eval_oracle': np.max(self.episode_reward_buffer_train_oracle),
                                                       'test':np.max(self.episode_reward_buffer_test),
                                                       'test_oracle':np.max(self.episode_reward_buffer_test_oracle),
                                                       'test_nondet':np.max(self.episode_reward_buffer_test_nondet)}
        episode_statistics['Rewards/mean_episodes'] = {'train': np.mean(self.episode_reward_buffer),
                                                       'train_eval': np.mean(self.episode_reward_buffer_train),
                                                       'train_eval_oracle': np.mean(self.episode_reward_buffer_train_oracle),
                                                       'test': np.mean(self.episode_reward_buffer_test),
                                                       'test_oracle': np.mean(self.episode_reward_buffer_test_oracle),
                                                       'test_nondet': np.mean(self.episode_reward_buffer_test_nondet)}
        episode_statistics['Rewards/min_episodes']  = {'train': np.min(self.episode_reward_buffer),
                                                       'train_eval': np.min(self.episode_reward_buffer_train),
                                                       'train_eval_oracle': np.min(self.episode_reward_buffer_train_oracle),
                                                       'test': np.min(self.episode_reward_buffer_test),
                                                       'test_oracle': np.min(self.episode_reward_buffer_test_oracle),
                                                       'test_nondet': np.min(self.episode_reward_buffer_test_nondet)}

        episode_statistics['Len/max_episodes']  = {'train': np.max(self.episode_len_buffer),
                                                   'train_eval': np.max(self.episode_len_buffer_train),
                                                   'train_eval_oracle': np.max(self.episode_len_buffer_train_oracle),
                                                   'test': np.max(self.episode_len_buffer_test),
                                                   'test_oracle': np.max(self.episode_len_buffer_test_oracle),
                                                   'test_nondet': np.max(self.episode_len_buffer_test_nondet)}
        episode_statistics['Len/mean_episodes'] = {'train': np.mean(self.episode_len_buffer),
                                                   'train_eval': np.mean(self.episode_len_buffer_train),
                                                   'train_eval_oracle': np.mean(self.episode_len_buffer_train_oracle),
                                                   'test': np.mean(self.episode_len_buffer_test),
                                                   'test_oracle': np.mean(self.episode_len_buffer_test_oracle),
                                                   'test_nondet': np.mean(self.episode_len_buffer_test_nondet)}
        episode_statistics['Len/min_episodes']  = {'train': np.min(self.episode_len_buffer),
                                                   'train_eval': np.min(self.episode_len_buffer_train),
                                                   'train_eval_oracle': np.min(self.episode_len_buffer_train_oracle),
                                                   'test': np.min(self.episode_len_buffer_test),
                                                   'test_oracle': np.min(self.episode_len_buffer_test_oracle),
                                                   'test_nondet': np.min(self.episode_len_buffer_test_nondet)}
        if len(self.episode_reward_buffer_val) > 0:
            episode_statistics['Rewards/max_episodes']['validation'] =  np.max(self.episode_reward_buffer_val)
            episode_statistics['Rewards/mean_episodes']['validation'] = np.mean(self.episode_reward_buffer_val)
            episode_statistics['Rewards/min_episodes']['validation'] = np.min(self.episode_reward_buffer_val)
            episode_statistics['Len/max_episodes']['validation'] = np.max(self.episode_len_buffer_val)
            episode_statistics['Len/mean_episodes']['validation'] = np.mean(self.episode_len_buffer_val)
            episode_statistics['Len/min_episodes']['validation'] = np.min(self.episode_len_buffer_val)

            episode_statistics['Rewards/max_episodes']['train_partial'] =  np.max(self.episode_reward_buffer_train_val)
            episode_statistics['Rewards/mean_episodes']['train_partial'] = np.mean(self.episode_reward_buffer_train_val)
            episode_statistics['Rewards/min_episodes']['train_partial'] = np.min(self.episode_reward_buffer_train_val)
            episode_statistics['Len/max_episodes']['train_partial'] = np.max(self.episode_len_buffer_train_val)
            episode_statistics['Len/mean_episodes']['train_partial'] = np.mean(self.episode_len_buffer_train_val)
            episode_statistics['Len/min_episodes']['train_partial'] = np.min(self.episode_len_buffer_train_val)

        return episode_statistics

    def get_train_val_statistics(self):
        train_statistics = {}
        train_statistics['Rewards_max_episodes'] = np.max(self.episode_reward_buffer)

        train_statistics['Rewards_mean_episodes'] = np.mean(self.episode_reward_buffer)

        train_statistics['Rewards_median_episodes'] = np.median(self.episode_reward_buffer)

        train_statistics['Rewards_min_episodes'] = np.min(self.episode_reward_buffer)

        train_statistics['Len_max_episodes'] = np.max(self.episode_len_buffer)

        train_statistics['Len_mean_episodes'] = np.mean(self.episode_len_buffer)

        train_statistics['Len_min_episodes'] = np.min(self.episode_len_buffer)

        if len(self.episode_reward_buffer_val)>0:
            train_statistics['Rewards_max_episodes_val'] = np.max(self.episode_reward_buffer_val)

            train_statistics['Rewards_mean_episodes_val'] = np.mean(self.episode_reward_buffer_val)

            train_statistics['Rewards_median_episodes_val'] = np.median(self.episode_reward_buffer_val)

            train_statistics['Rewards_min_episodes_val'] = np.min(self.episode_reward_buffer_val)

            train_statistics['Len_max_episodes_val'] = np.max(self.episode_len_buffer_val)

            train_statistics['Len_mean_episodes_val'] = np.mean(self.episode_len_buffer_val)

            train_statistics['Len_min_episodes_val'] = np.min(self.episode_len_buffer_val)

        return train_statistics
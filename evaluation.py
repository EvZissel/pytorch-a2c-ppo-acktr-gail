import numpy as np
import torch

from a2c_ppo_acktr import utils
# from a2c_ppo_acktr.envs import make_vec_envs
import matplotlib.pyplot as plt
from a2c_ppo_acktr.distributions import FixedCategorical
from torch import nn


def evaluate(actor_critic, obs_rms, eval_envs_dic, env_name, seed, num_processes, num_tasks, eval_log_dir,
             device, **kwargs):
    eval_envs = eval_envs_dic[env_name]
    eval_episode_rewards = []

    for iter in range(0, num_tasks, num_processes):
        for i in range(num_processes):
            eval_envs.set_task_id(task_id=iter + i, indices=i)
        vec_norm = utils.get_vec_normalize(eval_envs)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.obs_rms = obs_rms

        obs = eval_envs.reset()
        eval_recurrent_hidden_states = torch.zeros(
            num_processes, actor_critic.recurrent_hidden_state_size, device=device)
        eval_masks = torch.zeros(num_processes, 1, device=device)
        eval_attn_masks = torch.zeros(num_processes, 8, device=device)

        # while len(eval_episode_rewards) < 1:
        for t in range(kwargs["steps"]):
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states, eval_attn_masks = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    attn_masks=eval_attn_masks,
                    deterministic=True)

            # Obser reward and next obs
            obs, _, done, infos = eval_envs.step(action)
            # if len(eval_episode_rewards) > 98:
            #     print(action)
            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])
    # eval_envs.close()

    # print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
    #     len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    return eval_episode_rewards


def evaluate_procgen(actor_critic, eval_envs_dic, env_name, num_processes,
                     device, steps, logger, attention_features=False, det_masks=False, deterministic=True):
    eval_envs = eval_envs_dic[env_name]
    rew_batch = []
    done_batch = []
    seed_batch = []
    # eval_episode_len = []
    # eval_episode_len_buffer = []
    # for _ in range(num_processes):
    #     eval_episode_len_buffer.append(0)

    # obs = eval_envs.reset()
    # eval_recurrent_hidden_states = torch.zeros(
    #     num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    # eval_masks = torch.zeros(num_processes, 1, device=device)
    if attention_features:
        # eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        # eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        # eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        # eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

        eval_attn_masks = (torch.sigmoid(actor_critic.base.linear_attention) > 0.5).float()
        eval_attn_masks1 = (torch.sigmoid(actor_critic.base.block1.attention) > 0.5).float()
        eval_attn_masks2 = (torch.sigmoid(actor_critic.base.block2.attention) > 0.5).float()
        eval_attn_masks3 = (torch.sigmoid(actor_critic.base.block3.attention) > 0.5).float()
    elif actor_critic.attention_size == 1:
        eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    else:
        eval_attn_masks = torch.zeros(num_processes, *actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    # fig = plt.figure(figsize=(20, 20))
    # columns = 5
    # rows = 5
    # for i in range(1, columns * rows + 1):
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(obs[i].transpose())
    # plt.show()

    for t in range(steps):
        with torch.no_grad():
            _, action, _, dist_probs, eval_recurrent_hidden_states, _, _, _, _ = actor_critic.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            # if deterministic:
            #     dist_probs[:, 1] += dist_probs[:, 0]
            #     dist_probs[:, 1] += dist_probs[:, 2]
            #     dist_probs[:, 0] = 0
            #     dist_probs[:, 2] = 0
            #
            #     dist_probs[:, 7] += dist_probs[:, 6]
            #     dist_probs[:, 7] += dist_probs[:, 8]
            #     dist_probs[:, 6] = 0
            #     dist_probs[:, 8] = 0
            #     pure_action = dist_probs.max(1)[1].unsqueeze(1)
            #     action = pure_action

            # Observe reward and next obs
            next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
            logger.eval_masks[env_name] = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            logger.eval_recurrent_hidden_states[env_name] = eval_recurrent_hidden_states

            if 'env_reward' in infos[0]:
                rew_batch.append([info['env_reward'] for info in infos])
            else:
                rew_batch.append(reward)
            done_batch.append(done)

            if t == 0:
                prev_seeds = np.zeros_like(reward)
                for i in range(len(done)):
                    prev_seeds[i] = infos[i]['prev_level_seed']
                seed_batch.append(prev_seeds)

            seeds = np.zeros_like(reward)
            for i in range(len(done)):
                seeds[i] = infos[i]['level_seed']

            seed_batch.append(seeds)
            # for i, info in enumerate(infos):
            #     eval_episode_len_buffer[i] += 1
            #     if done[i] == True:
            #         eval_episode_rewards.append(reward[i])
            #         eval_episode_len.append(eval_episode_len_buffer[i])
            #         eval_episode_len_buffer[i] = 0

            logger.obs[env_name] = next_obs

    rew_batch = np.array(rew_batch)
    done_batch = np.array(done_batch)
    seed_batch = np.array(seed_batch)

    return rew_batch, done_batch


def evaluate_procgen_maxEnt(actor_critic, eval_envs_dic, env_name, num_processes,
                            device, steps, logger, attention_features=False, det_masks=False, deterministic=True):
    eval_envs = eval_envs_dic[env_name]
    rew_batch = []
    int_rew_batch = []
    done_batch = []
    seed_batch = []
    # eval_episode_len = []
    # eval_episode_len_buffer = []
    # for _ in range(num_processes):
    #     eval_episode_len_buffer.append(0)

    # obs = eval_envs.reset()
    # obs_sum = obs
    # eval_recurrent_hidden_states = torch.zeros(
    #     num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    # eval_masks = torch.ones(num_processes, 1, device=device)
    if attention_features:
        # eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        # eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        # eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        # eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

        eval_attn_masks = (torch.sigmoid(actor_critic.base.linear_attention) > 0.5).float()
        eval_attn_masks1 = (torch.sigmoid(actor_critic.base.block1.attention) > 0.5).float()
        eval_attn_masks2 = (torch.sigmoid(actor_critic.base.block2.attention) > 0.5).float()
        eval_attn_masks3 = (torch.sigmoid(actor_critic.base.block3.attention) > 0.5).float()
    elif actor_critic.attention_size == 1:
        eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    else:
        eval_attn_masks = torch.zeros(num_processes, *actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    # fig = plt.figure(figsize=(20, 20))
    # columns = 5
    # rows = 5
    # for i in range(1, columns * rows + 1):
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(obs[i].transpose())
    # plt.show()

    for t in range(steps):
        with torch.no_grad():
            _, action, _, dist_probs, eval_recurrent_hidden_states, _, _, _, _ = actor_critic.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            # if deterministic:
            #     dist_probs[:, 1] += dist_probs[:, 0]
            #     dist_probs[:, 1] += dist_probs[:, 2]
            #     dist_probs[:, 0] = 0
            #     dist_probs[:, 2] = 0
            #
            #     dist_probs[:, 7] += dist_probs[:, 6]
            #     dist_probs[:, 7] += dist_probs[:, 8]
            #     dist_probs[:, 6] = 0
            #     dist_probs[:, 8] = 0
            #     pure_action = dist_probs.max(1)[1].unsqueeze(1)
            #     action = pure_action

            # Observe reward and next obs
            next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
            logger.eval_masks[env_name] = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            logger.eval_recurrent_hidden_states[env_name] = eval_recurrent_hidden_states

            # if 'env_reward' in infos[0]:
            #     rew_batch.append([info['env_reward'] for info in infos])
            # else:
            #     rew_batch.append(reward)
            if t == 0:
                prev_seeds = np.zeros_like(reward)
                for i in range(len(done)):
                    prev_seeds[i] = infos[i]['prev_level_seed']
                seed_batch.append(prev_seeds)

            seeds = np.zeros_like(reward)
            for i in range(len(done)):
                seeds[i] = infos[i]['level_seed']
                if done[i] == 1:
                    logger.obs_sum[env_name][i] = next_obs[i]
                    logger.last_action[env_name][i] = torch.tensor([7])

            int_reward = np.zeros_like(reward)
            next_obs_sum = logger.obs_sum[env_name] + next_obs
            for i in range(len(int_reward)):
                num_zero_obs_sum = (logger.obs_sum[env_name][i][0] == 0).sum()
                num_zero_next_obs_sum = (next_obs_sum[i][0] == 0).sum()
                if num_zero_next_obs_sum < num_zero_obs_sum:
                    int_reward[i] = 1

            rew_batch.append(reward)
            int_rew_batch.append(int_reward)
            done_batch.append(done)
            seed_batch.append(seeds)

            logger.obs[env_name] = next_obs
            logger.obs_sum[env_name] = next_obs_sum
            logger.last_action[env_name] = action

    rew_batch = np.array(rew_batch)
    int_rew_batch = np.array(int_rew_batch)
    done_batch = np.array(done_batch)
    seed_batch = np.array(seed_batch)
    # num_zero_obs_end = np.zeros_like(reward)
    # for i in range(len(reward)):
    #     if (obs_sum[i][0] == 0).sum() == 0:
    #         num_zero_obs_end[i]= 1

    return rew_batch, int_rew_batch, done_batch, seed_batch


def evaluate_procgen_maxEnt_miner(actor_critic, eval_envs_dic, env_name, num_processes,
                                  device, steps, logger, attention_features=False, det_masks=False, deterministic=True):
    eval_envs = eval_envs_dic[env_name]
    rew_batch = []
    int_rew_batch = []
    done_batch = []
    seed_batch = []
    # eval_episode_len = []
    # eval_episode_len_buffer = []
    # for _ in range(num_processes):
    #     eval_episode_len_buffer.append(0)

    indices_row = torch.tensor([3, 9, 16, 22, 28, 35, 41, 48, 54, 61])
    indices_cal = torch.tensor([3, 10, 16, 22, 28, 35, 42, 48, 54, 61])

    # obs = eval_envs.reset()
    # obs_sum = obs
    # eval_recurrent_hidden_states = torch.zeros(
    #     num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    # eval_masks = torch.ones(num_processes, 1, device=device)
    if attention_features:
        # eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        # eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        # eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        # eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

        eval_attn_masks = (torch.sigmoid(actor_critic.base.linear_attention) > 0.5).float()
        eval_attn_masks1 = (torch.sigmoid(actor_critic.base.block1.attention) > 0.5).float()
        eval_attn_masks2 = (torch.sigmoid(actor_critic.base.block2.attention) > 0.5).float()
        eval_attn_masks3 = (torch.sigmoid(actor_critic.base.block3.attention) > 0.5).float()
    elif actor_critic.attention_size == 1:
        eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    else:
        eval_attn_masks = torch.zeros(num_processes, *actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    # fig = plt.figure(figsize=(20, 20))
    # columns = 5
    # rows = 5
    # for i in range(1, columns * rows + 1):
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(obs[i].transpose())
    # plt.show()

    for t in range(steps):
        with torch.no_grad():
            _, action, _, dist_probs, eval_recurrent_hidden_states, _, _, _, _ = actor_critic.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            if deterministic:
                dist_probs[:, 1] += dist_probs[:, 0]
                dist_probs[:, 1] += dist_probs[:, 2]
                dist_probs[:, 0] = 0
                dist_probs[:, 2] = 0

                dist_probs[:, 7] += dist_probs[:, 6]
                dist_probs[:, 7] += dist_probs[:, 8]
                dist_probs[:, 6] = 0
                dist_probs[:, 8] = 0
                pure_action = dist_probs.max(1)[1].unsqueeze(1)
                action = pure_action

            # Observe reward and next obs
            next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
            logger.eval_masks[env_name] = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            logger.eval_recurrent_hidden_states[env_name] = eval_recurrent_hidden_states

            # if 'env_reward' in infos[0]:
            #     rew_batch.append([info['env_reward'] for info in infos])
            # else:
            #     rew_batch.append(reward)
            if t == 0:
                prev_seeds = np.zeros_like(reward)
                for i in range(len(done)):
                    prev_seeds[i] = infos[i]['prev_level_seed']
                seed_batch.append(prev_seeds)

            seeds = np.zeros_like(reward)
            for i in range(len(done)):
                seeds[i] = infos[i]['level_seed']
                # if done[i] == 1:
                #     logger.obs_sum[env_name][i] = next_obs[i].cpu()
                #     logger.last_action[env_name][i] = torch.tensor([7])

            # int_reward = np.zeros_like(reward)
            # next_obs_sum = logger.obs_sum[env_name] * next_obs.cpu()
            # for i in range(len(int_reward)):
            #     num_zero_obs_sum = (logger.obs_sum[env_name][i][0] == 0).sum()
            #     num_zero_next_obs_sum = (next_obs_sum[i][0] == 0).sum()
            #     if num_zero_next_obs_sum > num_zero_obs_sum:
            #         int_reward[i] = 1

            int_reward = np.zeros_like(reward)
            for i in range(len(reward)):
                dirt = (logger.obs[env_name][i] * (logger.obs[env_name][i][2] > 0.1) * (logger.obs[env_name][i][2] < 0.3) * (logger.obs[env_name][i][0] > 0.3))[0].cpu()
                next_dirt = (next_obs[i] * (next_obs[i][2] > 0.1) * (next_obs[i][2] < 0.3) * (next_obs[i][0] > 0.3))[0].cpu()

                dirt_ds = torch.index_select(dirt, 0, indices_row)
                dirt_ds = torch.index_select(dirt_ds, 1, indices_cal)
                next_dirt_ds = torch.index_select(next_dirt, 0, indices_row)
                next_dirt_ds = torch.index_select(next_dirt_ds, 1, indices_cal)

                if done[i] == 0:
                    num_dirt_obs_sum = (dirt_ds > 0).sum()
                    num_dirt_next_obs_sum = (next_dirt_ds > 0).sum()
                    if num_dirt_next_obs_sum < num_dirt_obs_sum:
                        int_reward[i] = 1

            rew_batch.append(reward)
            int_rew_batch.append(int_reward)
            done_batch.append(done)
            seed_batch.append(seeds)

            logger.obs[env_name] = next_obs
            # logger.obs_sum[env_name] = next_obs_sum
            logger.last_action[env_name] = action

    rew_batch = np.array(rew_batch)
    int_rew_batch = np.array(int_rew_batch)
    done_batch = np.array(done_batch)
    seed_batch = np.array(seed_batch)
    # num_zero_obs_end = np.zeros_like(reward)
    # for i in range(len(reward)):
    #     if (obs_sum[i][0] == 0).sum() == 0:
    #         num_zero_obs_end[i]= 1

    return rew_batch, int_rew_batch, done_batch, seed_batch


def maxEnt_oracle(obs_all, action):
    next_action = torch.tensor(action)
    for i in range(len(action)):
        obs = obs_all[i].cpu().numpy()
        action_i = action[i]
        new_action_i = np.array([7])

        min_r = np.nonzero((obs[1] == 1))[0].min()
        max_r = np.nonzero((obs[1] == 1))[0].max()
        middle_r = int(min_r + (max_r - min_r + 1) / 2)

        min_c = np.nonzero((obs[1] == 1))[1].min()
        max_c = np.nonzero((obs[1] == 1))[1].max()
        middle_c = int(min_c + (max_c - min_c + 1) / 2)

        if action_i == 7:
            if (max_r + 1 < 64) and obs[0][max_r + 1, middle_c] == 0:
                new_action_i = np.array([3])
            elif (max_c + 1 < 64) and obs[0][middle_r, max_c + 1] == 0:
                new_action_i = np.array([7])
            elif (min_r - 1 > 0) and obs[0][min_r - 1, middle_c] == 0:
                new_action_i = np.array([5])
            else:
                new_action_i = np.array([1])
        elif action_i == 5:
            if (max_c + 1 < 64) and obs[0][middle_r, max_c + 1] == 0:
                new_action_i = np.array([7])
            elif (min_r - 1 > 0) and obs[0][min_r - 1, middle_c] == 0:
                new_action_i = np.array([5])
            elif (min_c - 1 > 0) and obs[0][middle_r, min_c - 1] == 0:
                new_action_i = np.array([1])
            else:
                new_action_i = np.array([3])
        elif action_i == 3:
            if (min_c - 1 > 0) and obs[0][middle_r, min_c - 1] == 0:
                new_action_i = np.array([1])
            elif (max_r + 1 < 64) and obs[0][max_r + 1, middle_c] == 0:
                new_action_i = np.array([3])
            elif (max_c + 1 < 64) and obs[0][middle_r, max_c + 1] == 0:
                new_action_i = np.array([7])
            else:
                new_action_i = np.array([5])
        elif action_i == 1:
            if (min_r - 1 > 0) and obs[0][min_r - 1, middle_c] == 0:
                new_action_i = np.array([5])
            elif (min_c - 1 > 0) and obs[0][middle_r, min_c - 1] == 0:
                new_action_i = np.array([1])
            elif (max_r + 1 < 64) and obs[0][max_r + 1, middle_c] == 0:
                new_action_i = np.array([3])
            else:
                new_action_i = np.array([7])

        next_action[i] = torch.tensor(new_action_i)

    return next_action


def maxEnt_oracle_WOr(obs_all, action):
    next_action = torch.tensor(action)
    for i in range(len(action)):
        obs = obs_all[i].cpu().numpy()
        action_i = action[i]
        new_action_i = np.array([7])

        min_r = np.nonzero((obs[1] == 1))[0].min()
        max_r = np.nonzero((obs[1] == 1))[0].max()
        middle_r = int(min_r + (max_r - min_r + 1) / 2)

        min_c = np.nonzero((obs[1] == 1))[1].min()
        max_c = np.nonzero((obs[1] == 1))[1].max()
        middle_c = int(min_c + (max_c - min_c + 1) / 2)

        if action_i == 7:
            if (max_r + 1 < 64) and (obs[0][max_r + 1, middle_c] == 0 or obs[2][max_r + 1, middle_c] == 1):
                new_action_i = np.array([3])
            elif (max_c + 1 < 64) and (obs[0][middle_r, max_c + 1] == 0 or obs[2][middle_r, max_c + 1] == 1):
                new_action_i = np.array([7])
            elif (min_r - 1 > 0) and (obs[0][min_r - 1, middle_c] == 0 or obs[2][min_r - 1, middle_c] == 1):
                new_action_i = np.array([5])
            else:
                new_action_i = np.array([1])
        elif action_i == 5:
            if (max_c + 1 < 64) and (obs[0][middle_r, max_c + 1] == 0 or obs[2][middle_r, max_c + 1] == 1):
                new_action_i = np.array([7])
            elif (min_r - 1 > 0) and (obs[0][min_r - 1, middle_c] == 0 or obs[2][min_r - 1, middle_c] == 1):
                new_action_i = np.array([5])
            elif (min_c - 1 > 0) and (obs[0][middle_r, min_c - 1] == 0 or obs[2][middle_r, min_c - 1] == 1):
                new_action_i = np.array([1])
            else:
                new_action_i = np.array([3])
        elif action_i == 3:
            if (min_c - 1 > 0) and (obs[0][middle_r, min_c - 1] == 0 or obs[2][middle_r, min_c - 1] == 1):
                new_action_i = np.array([1])
            elif (max_r + 1 < 64) and (obs[0][max_r + 1, middle_c] == 0 or obs[2][max_r + 1, middle_c] == 1):
                new_action_i = np.array([3])
            elif (max_c + 1 < 64) and (obs[0][middle_r, max_c + 1] == 0 or obs[2][middle_r, max_c + 1] == 1):
                new_action_i = np.array([7])
            else:
                new_action_i = np.array([5])
        elif action_i == 1:
            if (min_r - 1 > 0) and (obs[0][min_r - 1, middle_c] == 0 or obs[2][min_r - 1, middle_c] == 1):
                new_action_i = np.array([5])
            elif (min_c - 1 > 0) and (obs[0][middle_r, min_c - 1] == 0 or obs[2][middle_r, min_c - 1] == 1):
                new_action_i = np.array([1])
            elif (max_r + 1 < 64) and (obs[0][max_r + 1, middle_c] == 0 or obs[2][max_r + 1, middle_c] == 1):
                new_action_i = np.array([3])
            else:
                new_action_i = np.array([7])

        next_action[i] = torch.tensor(new_action_i)

    return next_action


def maxEnt_oracle_left_WOr(obs_all, action):
    next_action = torch.tensor(action)
    for i in range(len(action)):
        obs = obs_all[i].cpu().numpy()
        action_i = action[i]
        new_action_i = np.array([7])

        min_r = np.nonzero((obs[1] == 1))[0].min()
        max_r = np.nonzero((obs[1] == 1))[0].max()
        middle_r = int(min_r + (max_r - min_r + 1) / 2)

        min_c = np.nonzero((obs[1] == 1))[1].min()
        max_c = np.nonzero((obs[1] == 1))[1].max()
        middle_c = int(min_c + (max_c - min_c + 1) / 2)

        if action_i == 7:
            if (min_r - 1 > 0) and (obs[0][min_r - 1, middle_c] == 0 or obs[2][min_r - 1, middle_c] == 1):
                new_action_i = np.array([5])
            elif (max_c + 1 < 64) and (obs[0][middle_r, max_c + 1] == 0 or obs[2][middle_r, max_c + 1] == 1):
                new_action_i = np.array([7])
            elif (max_r + 1 < 64) and (obs[0][max_r + 1, middle_c] == 0 or obs[2][max_r + 1, middle_c] == 1):
                new_action_i = np.array([3])
            else:
                new_action_i = np.array([1])
        elif action_i == 5:
            if (min_c - 1 > 0) and (obs[0][middle_r, min_c - 1] == 0 or obs[2][middle_r, min_c - 1] == 1):
                new_action_i = np.array([1])
            elif (min_r - 1 > 0) and (obs[0][min_r - 1, middle_c] == 0 or obs[2][min_r - 1, middle_c] == 1):
                new_action_i = np.array([5])
            elif (max_c + 1 < 64) and (obs[0][middle_r, max_c + 1] == 0 or obs[2][middle_r, max_c + 1] == 1):
                new_action_i = np.array([7])
            else:
                new_action_i = np.array([3])
        elif action_i == 3:
            if (max_c + 1 < 64) and (obs[0][middle_r, max_c + 1] == 0 or obs[2][middle_r, max_c + 1] == 1):
                new_action_i = np.array([7])
            elif (max_r + 1 < 64) and (obs[0][max_r + 1, middle_c] == 0 or obs[2][max_r + 1, middle_c] == 1):
                new_action_i = np.array([3])
            elif (min_c - 1 > 0) and (obs[0][middle_r, min_c - 1] == 0 or obs[2][middle_r, min_c - 1] == 1):
                new_action_i = np.array([1])
            else:
                new_action_i = np.array([5])
        elif action_i == 1:
            if (max_r + 1 < 64) and (obs[0][max_r + 1, middle_c] == 0 or obs[2][max_r + 1, middle_c] == 1):
                new_action_i = np.array([3])
            elif (min_c - 1 > 0) and (obs[0][middle_r, min_c - 1] == 0 or obs[2][middle_r, min_c - 1] == 1):
                new_action_i = np.array([1])
            elif (min_r - 1 > 0) and (obs[0][min_r - 1, middle_c] == 0 or obs[2][min_r - 1, middle_c] == 1):
                new_action_i = np.array([5])
            else:
                new_action_i = np.array([7])

        next_action[i] = torch.tensor(new_action_i)

    return next_action


def maxEnt_oracle_left(obs_all, action):
    next_action = torch.tensor(action)
    for i in range(len(action)):
        obs = obs_all[i].cpu().numpy()
        action_i = action[i]
        new_action_i = np.array([7])

        min_r = np.nonzero((obs[1] == 1))[0].min()
        max_r = np.nonzero((obs[1] == 1))[0].max()
        middle_r = int(min_r + (max_r - min_r + 1) / 2)

        min_c = np.nonzero((obs[1] == 1))[1].min()
        max_c = np.nonzero((obs[1] == 1))[1].max()
        middle_c = int(min_c + (max_c - min_c + 1) / 2)

        if action_i == 7:
            if (min_r - 1 > 0) and obs[0][min_r - 1, middle_c] == 0:
                new_action_i = np.array([5])
            elif (max_c + 1 < 64) and obs[0][middle_r, max_c + 1] == 0:
                new_action_i = np.array([7])
            elif (max_r + 1 < 64) and obs[0][max_r + 1, middle_c] == 0:
                new_action_i = np.array([3])
            else:
                new_action_i = np.array([1])
        elif action_i == 5:
            if (min_c - 1 > 0) and obs[0][middle_r, min_c - 1] == 0:
                new_action_i = np.array([1])
            elif (min_r - 1 > 0) and obs[0][min_r - 1, middle_c] == 0:
                new_action_i = np.array([5])
            elif (max_c + 1 < 64) and obs[0][middle_r, max_c + 1] == 0:
                new_action_i = np.array([7])
            else:
                new_action_i = np.array([3])
        elif action_i == 3:
            if (max_c + 1 < 64) and obs[0][middle_r, max_c + 1] == 0:
                new_action_i = np.array([7])
            elif (max_r + 1 < 64) and obs[0][max_r + 1, middle_c] == 0:
                new_action_i = np.array([3])
            elif (min_c - 1 > 0) and obs[0][middle_r, min_c - 1] == 0:
                new_action_i = np.array([1])
            else:
                new_action_i = np.array([5])
        elif action_i == 1:
            if (max_r + 1 < 64) and obs[0][max_r + 1, middle_c] == 0:
                new_action_i = np.array([3])
            elif (min_c - 1 > 0) and obs[0][middle_r, min_c - 1] == 0:
                new_action_i = np.array([1])
            elif (min_r - 1 > 0) and obs[0][min_r - 1, middle_c] == 0:
                new_action_i = np.array([5])
            else:
                new_action_i = np.array([7])

        next_action[i] = torch.tensor(new_action_i)

    return next_action


def oracle_left(obs_all, action):
    next_action = torch.tensor(action)
    for i in range(len(action)):
        next_action[i] = torch.tensor(np.array([1]))

    return next_action


def oracle_right(obs_all, action):
    next_action = torch.tensor(action)
    for i in range(len(action)):
        next_action[i] = torch.tensor(np.array([7]))

    return next_action


def oracle_up(obs_all, action):
    next_action = torch.tensor(action)
    for i in range(len(action)):
        next_action[i] = torch.tensor(np.array([5]))

    return next_action


def oracle_down(obs_all, action):
    next_action = torch.tensor(action)
    for i in range(len(action)):
        next_action[i] = torch.tensor(np.array([3]))

    return next_action


def evaluate_procgen_LEEP(actor_critic_0, actor_critic_1, actor_critic_2, actor_critic_3, eval_envs_dic, env_name,
                          num_processes,
                          device, steps, logger, attention_features=False, det_masks=False, deterministic=True, num_ensemble=4,
                          actor_critic_4=None, actor_critic_5=None, actor_critic_6=None, actor_critic_7=None, actor_critic_8=None, actor_critic_9=None):
    eval_envs = eval_envs_dic[env_name]
    rew_batch = []
    int_rew_batch = []
    done_batch = []
    seed_batch = []
    # eval_episode_len = []
    # eval_episode_len_buffer = []
    # for _ in range(num_processes):
    #     eval_episode_len_buffer.append(0)

    # obs = eval_envs.reset()
    # obs_sum = obs
    # eval_recurrent_hidden_states = torch.zeros(
    #     num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    # eval_masks = torch.ones(num_processes, 1, device=device)
    if attention_features:
        # eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        # eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        # eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        # eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

        eval_attn_masks = (torch.sigmoid(actor_critic_0.base.linear_attention) > 0.5).float()
        eval_attn_masks1 = (torch.sigmoid(actor_critic_0.base.block1.attention) > 0.5).float()
        eval_attn_masks2 = (torch.sigmoid(actor_critic_0.base.block2.attention) > 0.5).float()
        eval_attn_masks3 = (torch.sigmoid(actor_critic_0.base.block3.attention) > 0.5).float()
    elif actor_critic_0.attention_size == 1:
        eval_attn_masks = torch.zeros(num_processes, actor_critic_0.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    else:
        eval_attn_masks = torch.zeros(num_processes, *actor_critic_0.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    # fig = plt.figure(figsize=(20, 20))
    # columns = 5
    # rows = 5
    # for i in range(1, columns * rows + 1):
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(obs[i].transpose())
    # plt.show()

    for t in range(steps):
        with torch.no_grad():
            _, action0, _, dist_probs, eval_recurrent_hidden_states, _, _, _, _ = actor_critic_0.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            _, action_1, _, dist_probs_1, eval_recurrent_hidden_states_1, _, _, _, _ = actor_critic_1.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            _, action_2, _, dist_probs_2, eval_recurrent_hidden_states_2, _, _, _, _ = actor_critic_2.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            _, action_3, _, dist_probs_3, eval_recurrent_hidden_states_3, _, _, _, _ = actor_critic_3.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            if num_ensemble > 4:
                _, action_4, _, dist_probs_4, eval_recurrent_hidden_states_4, _, _, _, _ = actor_critic_4.act(
                    logger.obs[env_name].float().to(device),
                    logger.eval_recurrent_hidden_states[env_name],
                    logger.eval_masks[env_name],
                    attn_masks=eval_attn_masks,
                    attn_masks1=eval_attn_masks1,
                    attn_masks2=eval_attn_masks2,
                    attn_masks3=eval_attn_masks3,
                    deterministic=deterministic,
                    reuse_masks=det_masks)

                _, action_5, _, dist_probs_5, eval_recurrent_hidden_states_5, _, _, _, _ = actor_critic_5.act(
                    logger.obs[env_name].float().to(device),
                    logger.eval_recurrent_hidden_states[env_name],
                    logger.eval_masks[env_name],
                    attn_masks=eval_attn_masks,
                    attn_masks1=eval_attn_masks1,
                    attn_masks2=eval_attn_masks2,
                    attn_masks3=eval_attn_masks3,
                    deterministic=deterministic,
                    reuse_masks=det_masks)

                if num_ensemble > 6:
                    _, action_6, _, dist_probs_6, eval_recurrent_hidden_states_6, _, _, _, _ = actor_critic_6.act(
                        logger.obs[env_name].float().to(device),
                        logger.eval_recurrent_hidden_states[env_name],
                        logger.eval_masks[env_name],
                        attn_masks=eval_attn_masks,
                        attn_masks1=eval_attn_masks1,
                        attn_masks2=eval_attn_masks2,
                        attn_masks3=eval_attn_masks3,
                        deterministic=deterministic,
                        reuse_masks=det_masks)

                    _, action_7, _, dist_probs_7, eval_recurrent_hidden_states_7, _, _, _, _ = actor_critic_7.act(
                        logger.obs[env_name].float().to(device),
                        logger.eval_recurrent_hidden_states[env_name],
                        logger.eval_masks[env_name],
                        attn_masks=eval_attn_masks,
                        attn_masks1=eval_attn_masks1,
                        attn_masks2=eval_attn_masks2,
                        attn_masks3=eval_attn_masks3,
                        deterministic=deterministic,
                        reuse_masks=det_masks)

                if num_ensemble > 8:
                    _, action_8, _, dist_probs_8, eval_recurrent_hidden_states_8, _, _, _, _ = actor_critic_8.act(
                        logger.obs[env_name].float().to(device),
                        logger.eval_recurrent_hidden_states[env_name],
                        logger.eval_masks[env_name],
                        attn_masks=eval_attn_masks,
                        attn_masks1=eval_attn_masks1,
                        attn_masks2=eval_attn_masks2,
                        attn_masks3=eval_attn_masks3,
                        deterministic=deterministic,
                        reuse_masks=det_masks)

                    _, action_9, _, dist_probs_9, eval_recurrent_hidden_states_9, _, _, _, _ = actor_critic_9.act(
                        logger.obs[env_name].float().to(device),
                        logger.eval_recurrent_hidden_states[env_name],
                        logger.eval_masks[env_name],
                        attn_masks=eval_attn_masks,
                        attn_masks1=eval_attn_masks1,
                        attn_masks2=eval_attn_masks2,
                        attn_masks3=eval_attn_masks3,
                        deterministic=deterministic,
                        reuse_masks=det_masks)

            max_policy = torch.max(torch.max(torch.max(dist_probs, dist_probs_1), dist_probs_2), dist_probs_3)
            if num_ensemble > 4:
                max_policy = torch.max(torch.max(max_policy, dist_probs_4), dist_probs_5)
            if num_ensemble > 6:
                max_policy = torch.max(torch.max(max_policy, dist_probs_6), dist_probs_7)
            if num_ensemble > 8:
                max_policy = torch.max(torch.max(max_policy, dist_probs_8), dist_probs_9)
            max_policy = torch.div(max_policy, max_policy.sum(1).unsqueeze(1))

            if deterministic:
                action = max_policy.max(1)[1]
            else:
                x = FixedCategorical(logits=max_policy)
                action = x.sample()
            # Observe reward and next obs
            next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
            logger.eval_masks[env_name] = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            logger.eval_recurrent_hidden_states[env_name] = eval_recurrent_hidden_states

            # if 'env_reward' in infos[0]:
            #     rew_batch.append([info['env_reward'] for info in infos])
            # else:
            #     rew_batch.append(reward)
            if t == 0:
                prev_seeds = np.zeros_like(reward)
                for i in range(len(done)):
                    prev_seeds[i] = infos[i]['prev_level_seed']
                seed_batch.append(prev_seeds)

            seeds = np.zeros_like(reward)
            for i in range(len(done)):
                seeds[i] = infos[i]['level_seed']
                if done[i] == 1:
                    logger.obs_sum[env_name][i] = next_obs[i].cpu()

            int_reward = np.zeros_like(reward)
            next_obs_sum = logger.obs_sum[env_name] + next_obs
            # for i in range(len(int_reward)):
            #     num_zero_obs_sum = (logger.obs_sum[env_name][i][0] == 0).sum()
            #     num_zero_next_obs_sum = (next_obs_sum[i][0] == 0).sum()
            #     if num_zero_next_obs_sum < num_zero_obs_sum:
            #         int_reward[i] = 1

            rew_batch.append(reward)
            int_rew_batch.append(int_reward)
            done_batch.append(done)
            seed_batch.append(seeds)

            logger.obs[env_name] = next_obs
            logger.obs_sum[env_name] = next_obs_sum

    rew_batch = np.array(rew_batch)
    int_rew_batch = np.array(int_rew_batch)
    done_batch = np.array(done_batch)
    seed_batch = np.array(seed_batch)
    # num_zero_obs_end = np.zeros_like(reward)
    # for i in range(len(reward)):
    #     if (obs_sum[i][0] == 0).sum() == 0:
    #         num_zero_obs_end[i]= 1

    return rew_batch, int_rew_batch, done_batch, seed_batch


def evaluate_procgen_ensemble(actor_critic, actor_critic_1, actor_critic_2, actor_critic_3, actor_critic_4, actor_critic_5, actor_critic_6, actor_critic_7, actor_critic_8, actor_critic_9, actor_critic_maxEnt,
                              eval_envs_dic, env_name, num_processes,
                              device, steps, logger, attention_features=False, det_masks=False, deterministic=True,
                              num_detEnt=0, rand_act=False,num_ensemble=4, num_agree=4 ,maze_miner=False, num_agent=0):
    eval_envs = eval_envs_dic[env_name]
    rew_batch = []
    done_batch = []
    seed_batch = []
    # eval_episode_len = []
    # eval_episode_len_buffer = []
    # for _ in range(num_processes):
    #     eval_episode_len_buffer.append(0)

    # obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    # eval_masks = torch.zeros(num_processes, 1, device=device)

    if attention_features:
        # eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        # eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        # eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        # eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

        eval_attn_masks = (torch.sigmoid(actor_critic.base.linear_attention) > 0.5).float()
        eval_attn_masks1 = (torch.sigmoid(actor_critic.base.block1.attention) > 0.5).float()
        eval_attn_masks2 = (torch.sigmoid(actor_critic.base.block2.attention) > 0.5).float()
        eval_attn_masks3 = (torch.sigmoid(actor_critic.base.block3.attention) > 0.5).float()
    elif actor_critic.attention_size == 1:
        eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    else:
        eval_attn_masks = torch.zeros(num_processes, *actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    # fig = plt.figure(figsize=(20, 20))
    # columns = 5
    # rows = 5
    # for i in range(1, columns * rows + 1):
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(obs[i].transpose())
    # plt.show()
    beta = 0.5
    moving_average_prob = 0.0

    # step_count = torch.zeros(num_processes,1,device=device).fill_(11)
    # env_steps = torch.zeros(num_processes,1,device=device)
    is_novel = torch.ones(num_processes, 1, dtype=torch.bool, device=device)
    m = FixedCategorical(
        torch.tensor([0.55, 0.25, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]).repeat(num_processes, 1)) #worked for maze
    # m = FixedCategorical(torch.tensor(
    #     [0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
    #      1 - 14 * 0.067]).repeat(num_processes, 1))
    # m = FixedCategorical(torch.tensor([0.75, 0.15, 0.05, 0.05]).repeat(num_processes, 1))
    rand_action = FixedCategorical(torch.tensor(
        [0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
         1 - 14 * 0.067]).repeat(num_processes, 1))
    maxEnt_steps = torch.zeros(num_processes, 1, device=device)
    for t in range(steps):
        with torch.no_grad():
            _, action0, _, dist_probs, eval_recurrent_hidden_states, _, _, _, _ = actor_critic.act(
                logger.obs[env_name].float().to(device),
                torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            # dist_probs[:, 1] += dist_probs[:, 0]
            # dist_probs[:, 1] += dist_probs[:, 2]
            # dist_probs[:, 0] = 0
            # dist_probs[:, 2] = 0
            #
            # dist_probs[:, 7] += dist_probs[:, 6]
            # dist_probs[:, 7] += dist_probs[:, 8]
            # dist_probs[:, 6] = 0
            # dist_probs[:, 8] = 0
            # pure_action0 = dist_probs.max(1)[1].unsqueeze(1)
            # prob_pure_action = dist_probs.max(1)[0].unsqueeze(1)
            # if deterministic:
            #     action0 = pure_action0

            # moving_average_prob = (1 - beta) * moving_average_prob + beta * prob_pure_action
            # idex_prob = (prob_pure_action > 0.9)
            # moving_average_prob = moving_average_prob*idex_prob

            _, action1, _, dist_probs1, eval_recurrent_hidden_states1, _, _, _, _ = actor_critic_1.act(
                logger.obs[env_name].float().to(device),
                torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            # dist_probs1[:, 1] += dist_probs1[:, 0]
            # dist_probs1[:, 1] += dist_probs1[:, 2]
            # dist_probs1[:, 0] = 0
            # dist_probs1[:, 2] = 0
            #
            # dist_probs1[:, 7] += dist_probs1[:, 6]
            # dist_probs1[:, 7] += dist_probs1[:, 8]
            # dist_probs1[:, 6] = 0
            # dist_probs1[:, 8] = 0
            # pure_action1 = dist_probs1.max(1)[1].unsqueeze(1)
            # prob_pure_action1 = dist_probs1.max(1)[0].unsqueeze(1)
            # if deterministic:
            #     action1 = pure_action1

            _, action2, _, dist_probs2, eval_recurrent_hidden_states2, _, _, _, _ = actor_critic_2.act(
                logger.obs[env_name].float().to(device),
                torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            # dist_probs2[:, 1] += dist_probs2[:, 0]
            # dist_probs2[:, 1] += dist_probs2[:, 2]
            # dist_probs2[:, 0] = 0
            # dist_probs2[:, 2] = 0
            #
            # dist_probs2[:, 7] += dist_probs2[:, 6]
            # dist_probs2[:, 7] += dist_probs2[:, 8]
            # dist_probs2[:, 6] = 0
            # dist_probs2[:, 8] = 0
            # pure_action2 = dist_probs2.max(1)[1].unsqueeze(1)
            # prob_pure_action2 = dist_probs2.max(1)[0].unsqueeze(1)
            # if deterministic:
            #     action2 = pure_action2

            _, action3, _, dist_probs3, eval_recurrent_hidden_states3, _, _, _, _ = actor_critic_3.act(
                logger.obs[env_name].float().to(device),
                torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            # dist_probs3[:, 1] += dist_probs3[:, 0]
            # dist_probs3[:, 1] += dist_probs3[:, 2]
            # dist_probs3[:, 0] = 0
            # dist_probs3[:, 2] = 0
            #
            # dist_probs3[:, 7] += dist_probs3[:, 6]
            # dist_probs3[:, 7] += dist_probs3[:, 8]
            # dist_probs3[:, 6] = 0
            # dist_probs3[:, 8] = 0
            # pure_action3 = dist_probs3.max(1)[1].unsqueeze(1)
            # prob_pure_action3 = dist_probs3.max(1)[0].unsqueeze(1)
            # if deterministic:
            #     action3 = pure_action3

            _, action_maxEnt, _, dist_probs_maxEnt, eval_recurrent_hidden_states_maxEnt, _, _, _, _ = actor_critic_maxEnt.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states_maxEnt[env_name],
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            # dist_probs_maxEnt[:, 1] += dist_probs_maxEnt[:, 0]
            # dist_probs_maxEnt[:, 1] += dist_probs_maxEnt[:, 2]
            # dist_probs_maxEnt[:, 0] = 0
            # dist_probs_maxEnt[:, 2] = 0
            #
            # dist_probs_maxEnt[:, 7] += dist_probs_maxEnt[:, 6]
            # dist_probs_maxEnt[:, 7] += dist_probs_maxEnt[:, 8]
            # dist_probs_maxEnt[:, 6] = 0
            # dist_probs_maxEnt[:, 8] = 0
            # pure_action_maxEnt = dist_probs_maxEnt.max(1)[1].unsqueeze(1)
            # prob_pure_action_maxEnt = dist_probs_maxEnt.max(1)[0].unsqueeze(1)
            # if deterministic:
            #     action_maxEnt = pure_action_maxEnt

            # is_not_maxEnt = (pure_action == pure_action1) * (pure_action == pure_action2) * (prob_pure_action > 0.5) * (prob_pure_action1 > 0.5) * (prob_pure_action2 > 0.5)

            # env_steps = env_steps+1

            if num_ensemble > 4:
                _, action4, _, dist_probs4, eval_recurrent_hidden_states4, _, _, _, _ = actor_critic_4.act(
                    logger.obs[env_name].float().to(device),
                    torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                    logger.eval_masks[env_name],
                    attn_masks=eval_attn_masks,
                    attn_masks1=eval_attn_masks1,
                    attn_masks2=eval_attn_masks2,
                    attn_masks3=eval_attn_masks3,
                    deterministic=deterministic,
                    reuse_masks=det_masks)

                _, action5, _, dist_probs5, eval_recurrent_hidden_states5, _, _, _, _ = actor_critic_5.act(
                    logger.obs[env_name].float().to(device),
                    torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                    logger.eval_masks[env_name],
                    attn_masks=eval_attn_masks,
                    attn_masks1=eval_attn_masks1,
                    attn_masks2=eval_attn_masks2,
                    attn_masks3=eval_attn_masks3,
                    deterministic=deterministic,
                    reuse_masks=det_masks)

            if num_ensemble > 6:
                _, action6, _, dist_probs6, eval_recurrent_hidden_states6, _, _, _, _ = actor_critic_6.act(
                    logger.obs[env_name].float().to(device),
                    torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                    logger.eval_masks[env_name],
                    attn_masks=eval_attn_masks,
                    attn_masks1=eval_attn_masks1,
                    attn_masks2=eval_attn_masks2,
                    attn_masks3=eval_attn_masks3,
                    deterministic=deterministic,
                    reuse_masks=det_masks)

                _, action7, _, dist_probs7, eval_recurrent_hidden_states7, _, _, _, _ = actor_critic_7.act(
                    logger.obs[env_name].float().to(device),
                    torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                    logger.eval_masks[env_name],
                    attn_masks=eval_attn_masks,
                    attn_masks1=eval_attn_masks1,
                    attn_masks2=eval_attn_masks2,
                    attn_masks3=eval_attn_masks3,
                    deterministic=deterministic,
                    reuse_masks=det_masks)

            if num_ensemble > 8:
                _, action8, _, dist_probs8, eval_recurrent_hidden_states8, _, _, _, _ = actor_critic_8.act(
                    logger.obs[env_name].float().to(device),
                    torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                    logger.eval_masks[env_name],
                    attn_masks=eval_attn_masks,
                    attn_masks1=eval_attn_masks1,
                    attn_masks2=eval_attn_masks2,
                    attn_masks3=eval_attn_masks3,
                    deterministic=deterministic,
                    reuse_masks=det_masks)

                _, action9, _, dist_probs9, eval_recurrent_hidden_states9, _, _, _, _ = actor_critic_9.act(
                    logger.obs[env_name].float().to(device),
                    torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                    logger.eval_masks[env_name],
                    attn_masks=eval_attn_masks,
                    attn_masks1=eval_attn_masks1,
                    attn_masks2=eval_attn_masks2,
                    attn_masks3=eval_attn_masks3,
                    deterministic=deterministic,
                    reuse_masks=det_masks)

            # actions_vec = torch.zeros([num_processes, eval_envs.action_space.n], device=device)
            # for i in range(num_processes):
            #     actions_vec[i, action0[i]] += 1
            #     actions_vec[i, action1[i]] += 1
            #     actions_vec[i, action2[i]] += 1
            #     actions_vec[i, action3[i]] += 1
            #     if num_ensemble > 4:
            #         actions_vec[i, action4[i]] += 1
            #         actions_vec[i, action5[i]] += 1
            #     if num_ensemble > 6:
            #         actions_vec[i, action6[i]] += 1
            #         actions_vec[i, action7[i]] += 1
            #     if num_ensemble > 8:
            #         actions_vec[i, action8[i]] += 1
            #         actions_vec[i, action9[i]] += 1
            #
            # actions_max = actions_vec.max(1)
            # cardinal_value = actions_max[0]
            # cardinal_index = actions_max[1]
            # is_equal = (cardinal_value >= num_agree).unsqueeze(1)
            # # action_NN = cardinal_index.unsqueeze(1)
            # action_NN = action0

            actions = []
            actions.append(action0)
            actions.append(action1)
            actions.append(action2)
            actions.append(action3)
            cardinal_left = 1*(action0 == 0)+1*(action0 == 1) + 1*(action0 == 2) + 1*(action1 == 0)+1*(action1 == 1) + 1*(action1 == 2) + 1*(action2 == 0)+1*(action2 == 1) + 1*(action2 == 2)\
                            + 1 * (action3 == 0) + 1 * (action3 == 1) + 1 * (action3 == 2)
            cardinal_right  = 1*(action0 == 6)+1*(action0 == 7) + 1*(action0 == 8) + 1*(action1 == 6)+1*(action1 == 7) + 1*(action1 == 8) + 1*(action2 == 6)+1*(action2 == 7) + 1*(action2 == 8)\
                            + 1 * (action3 == 6) + 1 * (action3 == 7) + 1 * (action3 == 8)
            if (maze_miner):  #maze and miner do not have right down/up left down/up
                cardinal_down = 1 * (action0 == 3) + 1 * (action1 == 3) + 1 * (action2 == 3) + 1 * (action3 == 3)
                cardinal_up = 1 * (action0 == 5) + 1 * (action1 == 5) + 1 * (action2 == 5) + 1 * (action3 == 5)
            else:
                cardinal_down  = 1*(action0 == 3) + 1*(action1 == 3) + 1*(action2 == 3) + 1*(action3 == 3) + 1*(action0 == 0) + 1*(action1 == 0) + 1*(action2 == 0) + 1*(action3 == 0)\
                                + 1*(action0 == 6) + 1*(action1 == 6) + 1*(action2 == 6) + 1*(action3 == 6)
                cardinal_up  = 1*(action0 == 5) + 1*(action1 == 5) + 1*(action2 == 5) + 1*(action3 == 5) + 1*(action0 == 2) + 1*(action1 == 2) + 1*(action2 == 2) + 1*(action3 == 2) \
                               + 1 * (action0 == 8) + 1 * (action1 == 8) + 1 * (action2 == 8) + 1 * (action3 == 8)
                cardinal_fire  = 1*(action0 == 9) + 1*(action1 == 9) + 1*(action2 == 9) + 1*(action3 == 9)
                cardinal_else  = 1*(action0 == 4) + 1*(action0 == 10) + 1*(action0 == 11) + 1*(action0 == 12) + 1*(action0 == 13) + 1*(action0 == 14) \
                               + 1*(action1 == 9) + 1*(action1 == 10) + 1*(action1 == 11) + 1*(action1 == 12) + 1*(action1 == 13) + 1*(action1 == 14)  \
                               + 1*(action2 == 9) + 1*(action2 == 10) + 1*(action2 == 11) + 1*(action2 == 12) + 1*(action2 == 13) + 1*(action2 == 14)  \
                               + 1*(action3 == 9) + 1*(action3 == 10) + 1*(action3 == 11) + 1*(action3 == 12) + 1*(action3 == 13) + 1*(action3 == 14)


            if num_ensemble > 4:
                actions.append(action4)
                actions.append(action5)
                cardinal_left += 1 * (action4 == 0) + 1 * (action4 == 1) + 1 * (action4 == 2) + 1 * (action5 == 0) + 1 * (action5 == 1) + 1 * (action5 == 2)
                cardinal_right += 1 * (action4 == 6) + 1 * (action4 == 7) + 1 * (action4 == 8) + 1 * (action5 == 6) + 1 * (action5 == 7) + 1 * (action5 == 8)
                if (maze_miner):
                    cardinal_down += 1 * (action4 == 3) + 1 * (action5 == 3)
                    cardinal_up += 1 * (action4 == 5) + 1 * (action5 == 5)
                else:
                    cardinal_down += 1 * (action4 == 3) + 1 * (action5 == 3) + 1 * (action4 == 0) + 1 * (action5 == 0) + 1 * (action4 == 6) + 1 * (action5 == 6)
                    cardinal_up += 1 * (action4 == 5) + 1 * (action5 == 5) + 1 * (action4 == 2) + 1 * (action5 == 2) + 1 * (action4 == 8) + 1 * (action5 == 8)
                    cardinal_fire += 1 * (action4 == 9) + 1 * (action5 == 9)
                    cardinal_else += 1 * (action4 == 4) + 1 * (action4 == 10) + 1 * (action4 == 11) + 1 * (action4 == 12) + 1 * (action4 == 13) + 1 * (action4 == 14) \
                                  + 1 * (action5 == 9) + 1 * (action5 == 10) + 1 * (action5 == 11) + 1 * (action5 == 12) + 1 * (action5 == 13) + 1 * (action5 == 14)


            if num_ensemble > 6:
                actions.append(action6)
                actions.append(action7)
                cardinal_left += 1 * (action6 == 0) + 1 * (action6 == 1) + 1 * (action6 == 2) + 1 * (action7 == 0) + 1 * (action7 == 1) + 1 * (action7 == 2)
                cardinal_right += 1 * (action6 == 6) + 1 * (action6 == 7) + 1 * (action6 == 8) + 1 * (action7 == 6) + 1 * (action7 == 7) + 1 * (action7 == 8)
                if (maze_miner):
                    cardinal_down += 1 * (action6 == 3) + 1 * (action7 == 3)
                    cardinal_up += 1 * (action6 == 5) + 1 * (action7 == 5)
                else:
                    cardinal_down += 1 * (action6 == 3) + 1 * (action7 == 3) + 1 * (action6 == 0) + 1 * (action7 == 0) + 1 * (action6 == 6) + 1 * (action7 == 6)
                    cardinal_up += 1 * (action6 == 5) + 1 * (action7 == 5) + 1 * (action6 == 2) + 1 * (action7 == 2) + 1 * (action6 == 8) + 1 * (action7 == 8)
                    cardinal_fire += 1 * (action6 == 9) + 1 * (action7 == 9)
                    cardinal_else += 1 * (action6 == 4) + 1 * (action6 == 10) + 1 * (action6 == 11) + 1 * (action6 == 12) + 1 * (action6 == 13) + 1 * (action6 == 14) \
                                  + 1 * (action7 == 9) + 1 * (action7 == 10) + 1 * (action7 == 11) + 1 * (action7 == 12) + 1 * (action7 == 13) + 1 * (action7 == 14)

            if num_ensemble > 8:
                actions.append(action8)
                actions.append(action9)
                cardinal_left += 1 * (action8 == 0) + 1 * (action8 == 1) + 1 * (action8 == 2) + 1 * (action9 == 0) + 1 * (action9 == 1) + 1 * (action9 == 2)
                cardinal_right += 1 * (action8 == 6) + 1 * (action8 == 7) + 1 * (action8 == 8) + 1 * (action9 == 6) + 1 * (action9 == 7) + 1 * (action9 == 8)
                if (maze_miner):
                    cardinal_down += 1 * (action8 == 3) + 1 * (action9 == 3)
                    cardinal_up += 1 * (action8 == 5) + 1 * (action9 == 5)
                else:
                    cardinal_down += 1 * (action8 == 3) + 1 * (action9 == 3) + 1 * (action8 == 0) + 1 * (action9 == 0) + 1 * (action8 == 6) + 1 * (action9 == 6)
                    cardinal_up += 1 * (action8 == 5) + 1 * (action9 == 5) + 1 * (action8 == 2) + 1 * (action9 == 2) + 1 * (action8 == 8) + 1 * (action9 == 8)
                    cardinal_fire += 1 * (action8 == 9) + 1 * (action9 == 9)
                    cardinal_else += 1 * (action8 == 4) + 1 * (action8 == 10) + 1 * (action8 == 11) + 1 * (action8 == 12) + 1 * (action8 == 13) + 1 * (action8 == 14) \
                                  + 1 * (action9 == 9) + 1 * (action9 == 10) + 1 * (action9 == 11) + 1 * (action9 == 12) + 1 * (action9 == 13) + 1 * (action9 == 14)


            if (maze_miner):
                directions = torch.cat((cardinal_up, cardinal_right, cardinal_down, cardinal_left), dim=1)
            else:
                directions = torch.cat((cardinal_up, cardinal_right, cardinal_down, cardinal_left, cardinal_fire, cardinal_else),dim=1)

            # cardinal_value = torch.max(directions, dim=1)[0]
            # cardinal_index = torch.max(directions, dim=1)[1].unsqueeze(1)

            action_cardinal_left = 1 * (actions[num_agent] == 0) + 1 * (actions[num_agent] == 1) + 1 * (actions[num_agent] == 2)
            action_cardinal_right = 1 * (actions[num_agent] == 6) + 1 * (actions[num_agent] == 7) + 1 * (actions[num_agent] == 8)
            if (maze_miner):
                action_cardinal_down = 1 * (actions[num_agent] == 3)
                action_cardinal_up = 1 * (actions[num_agent] == 5)
                action_directions = torch.cat((action_cardinal_up, action_cardinal_right, action_cardinal_down, action_cardinal_left), dim=1)
            else:
                action_cardinal_down = 1 * (actions[num_agent] == 3) + 1 * (actions[num_agent] == 0) + 1 * (actions[num_agent] == 6)
                action_cardinal_up = 1 * (actions[num_agent] == 5) + 1 * (actions[num_agent] == 2) + 1 * (actions[num_agent] == 8)
                action_cardinal_fire = 1 * (actions[num_agent] == 9)
                action_cardinal_else = 1 * (actions[num_agent] == 4) + 1 * (actions[num_agent] == 10) + 1 * (actions[num_agent] == 11) + 1 * (actions[num_agent] == 12) + 1 * (actions[num_agent] == 13) + 1 * (actions[num_agent] == 14)
                action_directions = torch.cat((action_cardinal_up, action_cardinal_right, action_cardinal_down, action_cardinal_left, action_cardinal_fire, action_cardinal_else), dim=1)

            action_cardinal_index = torch.max(action_directions, dim=1)[1]

            is_equal = (directions[torch.arange(32), action_cardinal_index] >= num_agree).unsqueeze(1)
            # is_equal = ((cardinal_index == action_cardinal_index) * (cardinal_value >= num_agree).unsqueeze(1))
            # is_equal =  (cardinal_value >= num_agree).unsqueeze(1)
            # lookup = torch.tensor([5, 7, 3, 1], device=device)
            # action_NN = lookup[cardinal_index].unsqueeze(1)
            # action_NN = action0
            action_NN = actions[num_agent]

            maxEnt_steps = maxEnt_steps - 1

            maxEnt_steps_sample = (~is_equal)*(maxEnt_steps<=0)
            maxEnt_steps = (m.sample() + 1).to(device)*maxEnt_steps_sample + maxEnt_steps*(~maxEnt_steps_sample)

            is_action = is_equal*(maxEnt_steps<=0)


            if num_detEnt > 0:
                maxEnt_steps = (num_detEnt * torch.ones(num_processes, 1, device=device)).to(device)*maxEnt_steps_sample + maxEnt_steps*(~maxEnt_steps_sample)

            action = action_NN * is_action + action_maxEnt * (~is_action)

            if rand_act:
                action = action_NN * is_action + rand_action.sample().to(device) * (~is_action)
            # action = pure_action*(~is_maxEnt) + pure_action_maxEnt*is_maxEnt
            # is_stuck = (env_steps > 100)
            # action = action*is_stuck + pure_action2*(~is_stuck)

            # is_not_maxEnt = (moving_average_prob > 0.9)
            # action = pure_action*is_not_maxEnt + pure_action_maxEnt*(~is_not_maxEnt)
            # action = pure_action2

            # max_policy = torch.max(torch.max(dist_probs, dist_probs1), dist_probs2)
            # max_policy = torch.div(max_policy, max_policy.sum(1).unsqueeze(1))

            # Observe reward and next obs
            # next_obs, reward, done, infos = eval_envs.step(max_policy.max(1)[1].squeeze().cpu().numpy())
            next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
            logger.eval_masks[env_name] = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            logger.eval_recurrent_hidden_states[env_name] = eval_recurrent_hidden_states
            logger.eval_recurrent_hidden_states1[env_name] = eval_recurrent_hidden_states1
            logger.eval_recurrent_hidden_states2[env_name] = eval_recurrent_hidden_states2
            logger.eval_recurrent_hidden_states_maxEnt[env_name] = eval_recurrent_hidden_states_maxEnt

            # if 'env_reward' in infos[0]:
            #     rew_batch.append([info['env_reward'] for info in infos])
            # else:
            #     rew_batch.append(reward)

            seeds = np.zeros_like(reward)
            # clean_reward = np.zeros_like(reward)
            # clean_done = np.zeros_like(reward)
            for i in range(len(done)):
                seeds[i] = infos[i]['level_seed']
                # if (t > 0 and np.array(done_batch)[:,i].sum() == 0):
                #     clean_reward[i] = reward[i]
                #     clean_done[i] = done[i]
            # seed_batch.append(seeds)
            # rew_batch.append(clean_reward)
            # done_batch.append(clean_done)

            seed_batch.append(seeds)
            rew_batch.append(reward)
            done_batch.append(done)

            # if t == 498:
            #     print("stop")
            #
            # is_novel = torch.zeros(num_processes, 1, dtype=torch.bool, device=device)
            # for i in range(len(done)):
            #     if done[i] == 1:
            #         logger.obs_sum[env_name][i] = next_obs[i].cpu()
            #         is_novel[i] = True
            #
            # next_obs_sum = logger.obs_sum[env_name] + next_obs.cpu()
            # for i in range(len(is_novel)):
            #     num_zero_obs_sum = (logger.obs_sum[env_name][i][0] == 0).sum()
            #     num_zero_next_obs_sum = (next_obs_sum[i][0] == 0).sum()
            #     if num_zero_next_obs_sum < num_zero_obs_sum:
            #         is_novel[i] = True

            # for i, info in enumerate(infos):
            #     eval_episode_len_buffer[i] += 1
            #     if done[i] == True:
            #         eval_episode_rewards.append(reward[i])
            #         eval_episode_len.append(eval_episode_len_buffer[i])
            #         eval_episode_len_buffer[i] = 0

            logger.obs[env_name] = next_obs
            # logger.obs_sum[env_name] = next_obs_sum

    rew_batch = np.array(rew_batch)
    done_batch = np.array(done_batch)
    seed_batch = np.array(seed_batch)
    # np.where(np.logical_xor(rew_batch, done_batch))
    # done_row_sum = done_batch.sum(0)
    # for i in range(len(done_row_sum)):
    #     if done_row_sum[i] == 0 :
    #         done_batch[steps-1,i] = True

    return rew_batch, done_batch


def evaluate_procgen_maxEnt_org(actor_critic, eval_envs_dic, env_name, num_processes,
                                device, steps, logger, eps_diff_NN, eps_NN, num_buffer, reset_cont,
                                attention_features=False, det_masks=False, deterministic=True):
    eval_envs = eval_envs_dic[env_name]
    rew_batch = []
    int_rew_batch = []
    done_batch = []
    seed_batch = []
    # eval_episode_len = []
    # eval_episode_len_buffer = []
    # for _ in range(num_processes):
    #     eval_episode_len_buffer.append(0)

    # obs = eval_envs.reset()
    # obs_sum = obs
    # eval_recurrent_hidden_states = torch.zeros(
    #     num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    # eval_masks = torch.ones(num_processes, 1, device=device)
    if attention_features:
        # eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        # eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        # eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        # eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

        eval_attn_masks = (torch.sigmoid(actor_critic.base.linear_attention) > 0.5).float()
        eval_attn_masks1 = (torch.sigmoid(actor_critic.base.block1.attention) > 0.5).float()
        eval_attn_masks2 = (torch.sigmoid(actor_critic.base.block2.attention) > 0.5).float()
        eval_attn_masks3 = (torch.sigmoid(actor_critic.base.block3.attention) > 0.5).float()
    elif actor_critic.attention_size == 1:
        eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    else:
        eval_attn_masks = torch.zeros(num_processes, *actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    # fig = plt.figure(figsize=(20, 20))
    # columns = 5
    # rows = 5
    # for i in range(1, columns * rows + 1):
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(obs[i].transpose())
    # plt.show()

    for t in range(steps):
        with torch.no_grad():
            _, action, _, dist_probs, eval_recurrent_hidden_states, _, _, _, _ = actor_critic.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            # Observe reward and next obs
            next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
            logger.eval_masks[env_name] = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            logger.eval_recurrent_hidden_states[env_name] = eval_recurrent_hidden_states

            # if 'env_reward' in infos[0]:
            #     rew_batch.append([info['env_reward'] for info in infos])
            # else:
            #     rew_batch.append(reward)
            if t == 0:
                prev_seeds = np.zeros_like(reward)
                for i in range(len(done)):
                    prev_seeds[i] = infos[i]['prev_level_seed']
                seed_batch.append(prev_seeds)

            seeds = np.zeros_like(reward)
            for i in range(len(done)):
                seeds[i] = infos[i]['level_seed']
                if done[i] == 1 or (logger.env_steps[env_name][i] % reset_cont == 0):
                    logger.obs_sum[env_name][i] = next_obs[i].cpu()
                    logger.obs0[env_name][i] = next_obs[i].cpu()
                    logger.env_steps[env_name][i] = 1
                    # logger.diff_obs[env_name][i] = []
                    # logger.diff_obs[env_name][i].append(torch.zeros_like(next_obs[i].cpu()))

            int_reward = np.zeros_like(reward)
            next_obs_sum = logger.obs_sum[env_name] + next_obs.cpu()
            diff_obs = torch.zeros_like(next_obs_sum)
            next_diff_obs = torch.zeros_like(next_obs_sum)
            for i in range(len(int_reward)):
                diff_obs[i] = 1 * (((logger.obs_sum[env_name][i] / logger.env_steps[env_name][i] -
                                     logger.obs0[env_name][i]) > eps_diff_NN) + ((logger.obs_sum[env_name][i] /
                                                                                  logger.env_steps[env_name][i] -
                                                                                  logger.obs0[env_name][
                                                                                      i]) < -eps_diff_NN))
                next_diff_obs[i] = 1 * (((next_obs_sum[i] / (logger.env_steps[env_name][i] + 1) - logger.obs0[env_name][
                    i]) > eps_diff_NN) + ((next_obs_sum[i] / (logger.env_steps[env_name][i] + 1) -
                                           logger.obs0[env_name][i]) < -eps_diff_NN))
                if done[i] == 0:
                    # ind = int(max(0, logger.env_steps[env_name][i] - num_buffer))
                    # diff_obs_swin = diff_obs[i] - logger.diff_obs[env_name][i][ind]
                    # next_diff_obs_swin = next_diff_obs[i] - logger.diff_obs[env_name][i][ind]
                    int_reward[i] = 1 * ((next_diff_obs[i] - diff_obs[i]).sum() > eps_NN)
                    logger.diff_obs[env_name][i].append(diff_obs[i])

            rew_batch.append(reward)
            int_rew_batch.append(int_reward)
            done_batch.append(done)
            seed_batch.append(seeds)

            logger.obs[env_name] = next_obs
            logger.obs_sum[env_name] = next_obs_sum
            logger.env_steps[env_name] = logger.env_steps[env_name] + 1

    rew_batch = np.array(rew_batch)
    int_rew_batch = np.array(int_rew_batch)
    done_batch = np.array(done_batch)
    seed_batch = np.array(seed_batch)
    # num_zero_obs_end = np.zeros_like(reward)
    # for i in range(len(reward)):
    #     if (obs_sum[i][0] == 0).sum() == 0:
    #         num_zero_obs_end[i]= 1

    return rew_batch, int_rew_batch, done_batch, seed_batch


def evaluate_procgen_maxEnt_L2(actor_critic, eval_envs_dic, env_name, num_processes,
                               device, steps, logger, eps_diff_NN, eps_NN, num_buffer, reset_cont,
                               attention_features=False, det_masks=False, deterministic=True):
    eval_envs = eval_envs_dic[env_name]
    rew_batch = []
    int_rew_batch = []
    done_batch = []
    seed_batch = []
    # eval_episode_len = []
    # eval_episode_len_buffer = []
    # for _ in range(num_processes):
    #     eval_episode_len_buffer.append(0)

    # obs = eval_envs.reset()
    # obs_sum = obs
    # eval_recurrent_hidden_states = torch.zeros(
    #     num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    # eval_masks = torch.ones(num_processes, 1, device=device)
    if attention_features:
        # eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        # eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        # eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        # eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

        eval_attn_masks = (torch.sigmoid(actor_critic.base.linear_attention) > 0.5).float()
        eval_attn_masks1 = (torch.sigmoid(actor_critic.base.block1.attention) > 0.5).float()
        eval_attn_masks2 = (torch.sigmoid(actor_critic.base.block2.attention) > 0.5).float()
        eval_attn_masks3 = (torch.sigmoid(actor_critic.base.block3.attention) > 0.5).float()
    elif actor_critic.attention_size == 1:
        eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    else:
        eval_attn_masks = torch.zeros(num_processes, *actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    # fig = plt.figure(figsize=(20, 20))
    # columns = 5
    # rows = 5
    # for i in range(1, columns * rows + 1):
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(obs[i].transpose())
    # plt.show()

    for t in range(steps):
        with torch.no_grad():
            _, action, _, dist_probs, eval_recurrent_hidden_states, _, _, _, _ = actor_critic.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            # Observe reward and next obs
            next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
            logger.eval_masks[env_name] = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            logger.eval_recurrent_hidden_states[env_name] = eval_recurrent_hidden_states

            # if 'env_reward' in infos[0]:
            #     rew_batch.append([info['env_reward'] for info in infos])
            # else:
            #     rew_batch.append(reward)
            if t == 0:
                prev_seeds = np.zeros_like(reward)
                for i in range(len(done)):
                    prev_seeds[i] = infos[i]['prev_level_seed']
                seed_batch.append(prev_seeds)

            seeds = np.zeros_like(reward)
            for i in range(len(done)):
                seeds[i] = infos[i]['level_seed']
                if done[i] == 1 or (logger.env_steps[env_name][i] % reset_cont == 0):
                    logger.env_steps[env_name][i] = 0
                    logger.obs_vec[env_name][i] = []

            int_reward = np.zeros_like(reward)
            for i in range(len(int_reward)):
                if done[i] == 0:
                    if len(logger.obs_vec[env_name][i]) > 0:
                        old_obs = torch.stack(logger.obs_vec[env_name][i])
                        norm2_dis = (old_obs - next_obs[i].unsqueeze(0)).reshape(int(logger.env_steps[env_name][i]),
                                                                                 -1).pow(2).sum(1)
                        int_reward[i] = 1 * (norm2_dis.min(0)[0] > eps_NN)

                logger.obs_vec[env_name][i].append(next_obs[i])

            rew_batch.append(reward)
            int_rew_batch.append(int_reward)
            done_batch.append(done)
            seed_batch.append(seeds)

            logger.obs[env_name] = next_obs
            # logger.obs_sum[env_name] = next_obs_sum
            logger.env_steps[env_name] = logger.env_steps[env_name] + 1

    rew_batch = np.array(rew_batch)
    int_rew_batch = np.array(int_rew_batch)
    done_batch = np.array(done_batch)
    seed_batch = np.array(seed_batch)
    # num_zero_obs_end = np.zeros_like(reward)
    # for i in range(len(reward)):
    #     if (obs_sum[i][0] == 0).sum() == 0:
    #         num_zero_obs_end[i]= 1

    return rew_batch, int_rew_batch, done_batch, seed_batch


def evaluate_procgen_maxEnt_avepool(actor_critic, eval_envs_dic, eval_envs_dic_full_obs, env_name, num_processes,
                                    device, steps, logger, kernel_size=3, stride=3, attention_features=False,
                                    det_masks=False, deterministic=True):
    eval_envs = eval_envs_dic[env_name]
    eval_envs_full_obs = eval_envs_dic_full_obs[env_name]
    rew_batch = []
    int_rew_batch = []
    done_batch = []
    seed_batch = []
    down_sample_avg = nn.AvgPool2d(kernel_size, stride=stride)
    # eval_episode_len = []
    # eval_episode_len_buffer = []
    # for _ in range(num_processes):
    #     eval_episode_len_buffer.append(0)

    # obs = eval_envs.reset()
    # obs_sum = obs
    # eval_recurrent_hidden_states = torch.zeros(
    #     num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    # eval_masks = torch.ones(num_processes, 1, device=device)
    if attention_features:
        # eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        # eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        # eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        # eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

        eval_attn_masks = (torch.sigmoid(actor_critic.base.linear_attention) > 0.5).float()
        eval_attn_masks1 = (torch.sigmoid(actor_critic.base.block1.attention) > 0.5).float()
        eval_attn_masks2 = (torch.sigmoid(actor_critic.base.block2.attention) > 0.5).float()
        eval_attn_masks3 = (torch.sigmoid(actor_critic.base.block3.attention) > 0.5).float()
    elif actor_critic.attention_size == 1:
        eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    else:
        eval_attn_masks = torch.zeros(num_processes, *actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    # fig = plt.figure(figsize=(20, 20))
    # columns = 5
    # rows = 5
    # for i in range(1, columns * rows + 1):
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(obs[i].transpose())
    # plt.show()

    for t in range(steps):
        with torch.no_grad():
            _, action, _, dist_probs, eval_recurrent_hidden_states, _, _, _, _ = actor_critic.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            # if deterministic:
            #     dist_probs[:, 1] += dist_probs[:, 0]
            #     dist_probs[:, 1] += dist_probs[:, 2]
            #     dist_probs[:, 0] = 0
            #     dist_probs[:, 2] = 0
            #
            #     dist_probs[:, 7] += dist_probs[:, 6]
            #     dist_probs[:, 7] += dist_probs[:, 8]
            #     dist_probs[:, 6] = 0
            #     dist_probs[:, 8] = 0
            #     pure_action = dist_probs.max(1)[1].unsqueeze(1)
            #     action = pure_action

            # Observe reward and next obs
            next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
            next_obs_full, _, _, _ = eval_envs_full_obs.step(action.squeeze().cpu().numpy())
            if kernel_size == 25:
                next_obs_full = torch.zeros_like(logger.obs_sum[env_name])
                next_obs_full_list = eval_envs_full_obs.env.get_info()
                for i in range(len(done)):
                    next_obs_full[i] = torch.tensor(next_obs_full_list[i]['rgb'] / 255).transpose(0, 2)

            logger.eval_masks[env_name] = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            logger.eval_recurrent_hidden_states[env_name] = eval_recurrent_hidden_states

            # if 'env_reward' in infos[0]:
            #     rew_batch.append([info['env_reward'] for info in infos])
            # else:
            #     rew_batch.append(reward)
            if t == 0:
                prev_seeds = np.zeros_like(reward)
                for i in range(len(done)):
                    prev_seeds[i] = infos[i]['prev_level_seed']
                seed_batch.append(prev_seeds)

            seeds = np.zeros_like(reward)
            for i in range(len(done)):
                seeds[i] = infos[i]['level_seed']
                if done[i] == 1:
                    logger.obs_sum[env_name][i] = next_obs_full[i]
                    # logger.last_action[env_name][i] = torch.tensor([7])

            int_reward = np.zeros_like(reward)
            next_obs_sum = logger.obs_sum[env_name] + next_obs_full
            next_obs_sum_ds = down_sample_avg(next_obs_sum)
            obs_sum_ds = down_sample_avg(logger.obs_sum[env_name])
            for i in range(len(int_reward)):
                num_zero_obs_sum = (obs_sum_ds[i][0] == 0).sum()
                num_zero_next_obs_sum = (next_obs_sum_ds[i][0] == 0).sum()
                int_reward[i] = num_zero_obs_sum - num_zero_next_obs_sum

            rew_batch.append(reward)
            int_rew_batch.append(int_reward)
            done_batch.append(done)
            seed_batch.append(seeds)

            logger.obs[env_name] = next_obs
            logger.obs_sum[env_name] = next_obs_sum
            logger.last_action[env_name] = action

    rew_batch = np.array(rew_batch)
    int_rew_batch = np.array(int_rew_batch)
    done_batch = np.array(done_batch)
    seed_batch = np.array(seed_batch)
    # num_zero_obs_end = np.zeros_like(reward)
    # for i in range(len(reward)):
    #     if (obs_sum[i][0] == 0).sum() == 0:
    #         num_zero_obs_end[i]= 1

    return rew_batch, int_rew_batch, done_batch, seed_batch


def evaluate_procgen_maxEnt_avepool_original(actor_critic, eval_envs_dic, eval_envs_dic_full_obs, env_name,
                                             num_processes,
                                             device, steps, logger, kernel_size=3, stride=3, attention_features=False,
                                             det_masks=False, deterministic=True):
    eval_envs = eval_envs_dic[env_name]
    eval_envs_full_obs = eval_envs_dic_full_obs[env_name]
    rew_batch = []
    int_rew_batch = []
    done_batch = []
    seed_batch = []
    down_sample_avg = nn.AvgPool2d(kernel_size, stride=stride)
    # eval_episode_len = []
    # eval_episode_len_buffer = []
    # for _ in range(num_processes):
    #     eval_episode_len_buffer.append(0)

    # obs = eval_envs.reset()
    # obs_sum = obs
    # eval_recurrent_hidden_states = torch.zeros(
    #     num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    # eval_masks = torch.ones(num_processes, 1, device=device)
    if attention_features:
        # eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        # eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        # eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        # eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

        eval_attn_masks = (torch.sigmoid(actor_critic.base.linear_attention) > 0.5).float()
        eval_attn_masks1 = (torch.sigmoid(actor_critic.base.block1.attention) > 0.5).float()
        eval_attn_masks2 = (torch.sigmoid(actor_critic.base.block2.attention) > 0.5).float()
        eval_attn_masks3 = (torch.sigmoid(actor_critic.base.block3.attention) > 0.5).float()
    elif actor_critic.attention_size == 1:
        eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    else:
        eval_attn_masks = torch.zeros(num_processes, *actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    # fig = plt.figure(figsize=(20, 20))
    # columns = 5
    # rows = 5
    # for i in range(1, columns * rows + 1):
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(obs[i].transpose())
    # plt.show()

    for t in range(steps):
        with torch.no_grad():
            _, action, _, dist_probs, eval_recurrent_hidden_states, _, _, _, _ = actor_critic.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            # if deterministic:
            #     dist_probs[:, 1] += dist_probs[:, 0]
            #     dist_probs[:, 1] += dist_probs[:, 2]
            #     dist_probs[:, 0] = 0
            #     dist_probs[:, 2] = 0
            #
            #     dist_probs[:, 7] += dist_probs[:, 6]
            #     dist_probs[:, 7] += dist_probs[:, 8]
            #     dist_probs[:, 6] = 0
            #     dist_probs[:, 8] = 0
            #     pure_action = dist_probs.max(1)[1].unsqueeze(1)
            #     action = pure_action

            # Observe reward and next obs
            next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
            next_obs_full, _, _, _ = eval_envs_full_obs.step(action.squeeze().cpu().numpy())
            if kernel_size == 25:
                next_obs_full = torch.zeros_like(logger.obs_sum[env_name])
                next_obs_full_list = eval_envs_full_obs.env.get_info()
                for i in range(len(done)):
                    next_obs_full[i] = torch.tensor(next_obs_full_list[i]['rgb'] / 255).transpose(0, 2)

            logger.eval_masks[env_name] = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            logger.eval_recurrent_hidden_states[env_name] = eval_recurrent_hidden_states

            # if 'env_reward' in infos[0]:
            #     rew_batch.append([info['env_reward'] for info in infos])
            # else:
            #     rew_batch.append(reward)
            if t == 0:
                prev_seeds = np.zeros_like(reward)
                for i in range(len(done)):
                    prev_seeds[i] = infos[i]['prev_level_seed']
                seed_batch.append(prev_seeds)

            seeds = np.zeros_like(reward)
            for i in range(len(done)):
                seeds[i] = infos[i]['level_seed']
                if done[i] == 1:
                    logger.obs_sum[env_name][i] = torch.zeros_like(next_obs_full[i].cpu())
                    logger.obs_full[env_name][i] = next_obs_full[i].cpu()
                    # logger.obs_sum[env_name][i] = next_obs_full[i].cpu()
                    # logger.last_action[env_name][i] = torch.tensor([7])

            int_reward = np.zeros_like(reward)
            # next_obs_sum = logger.obs_sum[env_name] + next_obs_full.cpu()
            # next_obs_sum_ds = down_sample_avg(next_obs_sum)
            # obs_sum_ds = down_sample_avg(logger.obs_sum[env_name])
            # for i in range(len(int_reward)):
            #     num_zero_obs_sum = (obs_sum_ds[i][0] == 0).sum()
            #     num_zero_next_obs_sum = (next_obs_sum_ds[i][0] == 0).sum()
            #     int_reward[i] = num_zero_obs_sum - num_zero_next_obs_sum
            next_obs_diff = 1 * ((next_obs_full - logger.obs_full[env_name].to(device)).abs() > 1e-5)
            next_obs_sum = logger.obs_sum[env_name].to(device) + next_obs_diff
            next_obs_sum = (1 * (down_sample_avg(next_obs_sum).abs() > 1e-5)).sum(1)
            obs_sum = (1 * (down_sample_avg(logger.obs_sum[env_name].to(device)).abs() > 1e-5)).sum(1)
            for i in range(len(done)):
                # int_reward[i] = (next_obs_sum[i] - obs_sum[i]).sum() / 3  # for RGB images
                num_zero_obs_sum = (obs_sum[i] == 0).sum()
                num_zero_next_obs_sum = (next_obs_sum[i] == 0).sum()
                int_reward[i] = num_zero_obs_sum - num_zero_next_obs_sum

            rew_batch.append(reward)
            int_rew_batch.append(int_reward)
            done_batch.append(done)
            seed_batch.append(seeds)

            logger.obs[env_name] = next_obs
            logger.obs_full[env_name] = next_obs_full
            logger.obs_sum[env_name] += next_obs_diff
            logger.last_action[env_name] = action

    rew_batch = np.array(rew_batch)
    int_rew_batch = np.array(int_rew_batch)
    done_batch = np.array(done_batch)
    seed_batch = np.array(seed_batch)
    # num_zero_obs_end = np.zeros_like(reward)
    # for i in range(len(reward)):
    #     if (obs_sum[i][0] == 0).sum() == 0:
    #         num_zero_obs_end[i]= 1

    return rew_batch, int_rew_batch, done_batch, seed_batch

def evaluate_procgen_maxEnt_avepool_original_L2(actor_critic, eval_envs_dic, eval_envs_dic_full_obs, env_name,
                                             num_processes,
                                             device, steps, logger, num_buffer, kernel_size=3, stride=3, attention_features=False,
                                             det_masks=False, deterministic=True, p_norm=2, neighbor_size=1):
    eval_envs = eval_envs_dic[env_name]
    eval_envs_full_obs = eval_envs_dic_full_obs[env_name]
    rew_batch = []
    int_rew_batch = []
    done_batch = []
    seed_batch = []
    down_sample_avg = nn.AvgPool2d(kernel_size, stride=stride)
    # eval_episode_len = []
    # eval_episode_len_buffer = []
    # for _ in range(num_processes):
    #     eval_episode_len_buffer.append(0)

    # obs = eval_envs.reset()
    # obs_sum = obs
    # eval_recurrent_hidden_states = torch.zeros(
    #     num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    # eval_masks = torch.ones(num_processes, 1, device=device)
    if attention_features:
        # eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        # eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        # eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        # eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

        eval_attn_masks = (torch.sigmoid(actor_critic.base.linear_attention) > 0.5).float()
        eval_attn_masks1 = (torch.sigmoid(actor_critic.base.block1.attention) > 0.5).float()
        eval_attn_masks2 = (torch.sigmoid(actor_critic.base.block2.attention) > 0.5).float()
        eval_attn_masks3 = (torch.sigmoid(actor_critic.base.block3.attention) > 0.5).float()
    elif actor_critic.attention_size == 1:
        eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    else:
        eval_attn_masks = torch.zeros(num_processes, *actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

    # fig = plt.figure(figsize=(20, 20))
    # columns = 5
    # rows = 5
    # for i in range(1, columns * rows + 1):
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(obs[i].transpose())
    # plt.show()

    for t in range(steps):
        with torch.no_grad():
            _, action, _, dist_probs, eval_recurrent_hidden_states, _, _, _, _ = actor_critic.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                attn_masks=eval_attn_masks,
                attn_masks1=eval_attn_masks1,
                attn_masks2=eval_attn_masks2,
                attn_masks3=eval_attn_masks3,
                deterministic=deterministic,
                reuse_masks=det_masks)

            # if deterministic:
            #     dist_probs[:, 1] += dist_probs[:, 0]
            #     dist_probs[:, 1] += dist_probs[:, 2]
            #     dist_probs[:, 0] = 0
            #     dist_probs[:, 2] = 0
            #
            #     dist_probs[:, 7] += dist_probs[:, 6]
            #     dist_probs[:, 7] += dist_probs[:, 8]
            #     dist_probs[:, 6] = 0
            #     dist_probs[:, 8] = 0
            #     pure_action = dist_probs.max(1)[1].unsqueeze(1)
            #     action = pure_action

            # Observe reward and next obs
            next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
            next_obs_full, _, _, _ = eval_envs_full_obs.step(action.squeeze().cpu().numpy())

            logger.eval_masks[env_name] = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            logger.eval_recurrent_hidden_states[env_name] = eval_recurrent_hidden_states

            # if 'env_reward' in infos[0]:
            #     rew_batch.append([info['env_reward'] for info in infos])
            # else:
            #     rew_batch.append(reward)
            if t == 0:
                prev_seeds = np.zeros_like(reward)
                for i in range(len(done)):
                    prev_seeds[i] = infos[i]['prev_level_seed']
                seed_batch.append(prev_seeds)

            seeds = np.zeros_like(reward)
            int_reward = np.zeros_like(reward)
            next_obs_ds = down_sample_avg(next_obs_full)
            for i in range(len(done)):
                seeds[i] = infos[i]['level_seed']
                if done[i] == 1 :
                    logger.obs_vec_ds[env_name][i] = []
                else:
                    env_steps = len(logger.obs_vec_ds[env_name][i])
                    if env_steps > 0:
                        if env_steps > num_buffer:
                            old_obs = torch.stack(logger.obs_vec_ds[env_name][i][env_steps-num_buffer:])
                        else:
                            old_obs = torch.stack(logger.obs_vec_ds[env_name][i])
                        neighbor_size_i = min(neighbor_size, len(logger.obs_vec_ds[env_name][i]))
                        int_reward[i]  = (old_obs - next_obs_ds[i].unsqueeze(0)).flatten(start_dim=1).norm(p=p_norm, dim=1).sort().values[int(neighbor_size_i - 1)]

                logger.obs_vec_ds[env_name][i].append(next_obs_ds[i])


            rew_batch.append(reward)
            int_rew_batch.append(int_reward)
            done_batch.append(done)
            seed_batch.append(seeds)

            logger.obs[env_name] = next_obs
            logger.obs_full[env_name] = next_obs_full
            logger.last_action[env_name] = action

    rew_batch = np.array(rew_batch)
    int_rew_batch = np.array(int_rew_batch)
    done_batch = np.array(done_batch)
    seed_batch = np.array(seed_batch)

    return rew_batch, int_rew_batch, done_batch, seed_batch
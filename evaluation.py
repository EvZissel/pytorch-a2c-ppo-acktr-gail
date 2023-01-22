import numpy as np
import torch

from a2c_ppo_acktr import utils
# from a2c_ppo_acktr.envs import make_vec_envs
import matplotlib.pyplot as plt
from a2c_ppo_acktr.distributions import FixedCategorical

def evaluate(actor_critic, obs_rms, eval_envs_dic, env_name, seed, num_processes, num_tasks, eval_log_dir,
             device, **kwargs):

    eval_envs = eval_envs_dic[env_name]
    eval_episode_rewards = []

    for iter in range(0, num_tasks, num_processes):
        for i in range(num_processes):
            eval_envs.set_task_id(task_id=iter+i, indices=i)
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
        eval_attn_masks1 = torch.zeros(num_processes,  16 , device=device)
        eval_attn_masks2 = torch.zeros(num_processes,  32 , device=device)
        eval_attn_masks3 = torch.zeros(num_processes,  32 , device=device)

    else:
        eval_attn_masks = torch.zeros(num_processes, *actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes,  16 , device=device)
        eval_attn_masks2 = torch.zeros(num_processes,  32 , device=device)
        eval_attn_masks3 = torch.zeros(num_processes,  32 , device=device)

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

            if 'env_reward' in infos[0]:
                rew_batch.append([info['env_reward'] for info in infos])
            else:
                rew_batch.append(reward)
            done_batch.append(done)

            # for i, info in enumerate(infos):
            #     eval_episode_len_buffer[i] += 1
            #     if done[i] == True:
            #         eval_episode_rewards.append(reward[i])
            #         eval_episode_len.append(eval_episode_len_buffer[i])
            #         eval_episode_len_buffer[i] = 0

            logger.obs[env_name] = next_obs

    rew_batch = np.array(rew_batch)
    done_batch = np.array(done_batch)

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
        eval_attn_masks1 = torch.zeros(num_processes,  16 , device=device)
        eval_attn_masks2 = torch.zeros(num_processes,  32 , device=device)
        eval_attn_masks3 = torch.zeros(num_processes,  32 , device=device)

    else:
        eval_attn_masks = torch.zeros(num_processes, *actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes,  16 , device=device)
        eval_attn_masks2 = torch.zeros(num_processes,  32 , device=device)
        eval_attn_masks3 = torch.zeros(num_processes,  32 , device=device)


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
            if t==0:
                prev_seeds = np.zeros_like(reward)
                for i in range(len(done)):
                    prev_seeds[i] = infos[i]['prev_level_seed']
                seed_batch.append(prev_seeds)

            seeds = np.zeros_like(reward)
            for i in range(len(done)):
                seeds[i] = infos[i]['level_seed']
                if done[i] == 1:
                    logger.obs_sum[env_name][i] = next_obs[i].cpu()
                    logger.last_action[env_name][i] = torch.tensor([7])

            int_reward = np.zeros_like(reward)
            next_obs_sum = logger.obs_sum[env_name] + next_obs.cpu()
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
        eval_attn_masks1 = torch.zeros(num_processes,  16 , device=device)
        eval_attn_masks2 = torch.zeros(num_processes,  32 , device=device)
        eval_attn_masks3 = torch.zeros(num_processes,  32 , device=device)

    else:
        eval_attn_masks = torch.zeros(num_processes, *actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes,  16 , device=device)
        eval_attn_masks2 = torch.zeros(num_processes,  32 , device=device)
        eval_attn_masks3 = torch.zeros(num_processes,  32 , device=device)


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
            if t==0:
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
                dirt = logger.obs[env_name][i] * (logger.obs[env_name][i][2] > 0.1) * (logger.obs[env_name][i][2] < 0.3) * (logger.obs[env_name][i][0] > 0.3)
                next_dirt = next_obs[i] * (next_obs[i][2] > 0.1) * (next_obs[i][2] < 0.3) * (next_obs[i][0] > 0.3)
                if done[i] == 0:
                    num_dirt_obs_sum = (dirt[0] > 0).sum()
                    num_dirt_next_obs_sum = (next_dirt[0] > 0).sum()
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
        obs =  obs_all[i].cpu().numpy()
        action_i = action[i]
        new_action_i = np.array([7])


        min_r = np.nonzero((obs[1] == 1))[0].min()
        max_r = np.nonzero((obs[1] == 1))[0].max()
        middle_r = int(min_r + (max_r - min_r + 1)/2)

        min_c = np.nonzero((obs[1] == 1))[1].min()
        max_c = np.nonzero((obs[1] == 1))[1].max()
        middle_c = int(min_c + (max_c - min_c + 1)/2)

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
        obs =  obs_all[i].cpu().numpy()
        action_i = action[i]
        new_action_i = np.array([7])


        min_r = np.nonzero((obs[1] == 1))[0].min()
        max_r = np.nonzero((obs[1] == 1))[0].max()
        middle_r = int(min_r + (max_r - min_r + 1)/2)

        min_c = np.nonzero((obs[1] == 1))[1].min()
        max_c = np.nonzero((obs[1] == 1))[1].max()
        middle_c = int(min_c + (max_c - min_c + 1)/2)

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
        obs =  obs_all[i].cpu().numpy()
        action_i = action[i]
        new_action_i = np.array([7])


        min_r = np.nonzero((obs[1] == 1))[0].min()
        max_r = np.nonzero((obs[1] == 1))[0].max()
        middle_r = int(min_r + (max_r - min_r + 1)/2)

        min_c = np.nonzero((obs[1] == 1))[1].min()
        max_c = np.nonzero((obs[1] == 1))[1].max()
        middle_c = int(min_c + (max_c - min_c + 1)/2)

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
        obs =  obs_all[i].cpu().numpy()
        action_i = action[i]
        new_action_i = np.array([7])


        min_r = np.nonzero((obs[1] == 1))[0].min()
        max_r = np.nonzero((obs[1] == 1))[0].max()
        middle_r = int(min_r + (max_r - min_r + 1)/2)

        min_c = np.nonzero((obs[1] == 1))[1].min()
        max_c = np.nonzero((obs[1] == 1))[1].max()
        middle_c = int(min_c + (max_c - min_c + 1)/2)

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
        next_action[i] = torch.tensor( np.array([1]))

    return next_action

def oracle_right(obs_all, action):
    next_action = torch.tensor(action)
    for i in range(len(action)):
        next_action[i] =  torch.tensor(np.array([7]))

    return next_action

def oracle_up(obs_all, action):
    next_action = torch.tensor(action)
    for i in range(len(action)):
        next_action[i] =  torch.tensor(np.array([5]))

    return next_action

def oracle_down(obs_all, action):
    next_action = torch.tensor(action)
    for i in range(len(action)):
        next_action[i] =  torch.tensor(np.array([3]))

    return next_action

def evaluate_procgen_LEEP(actor_critic_0, actor_critic_1, actor_critic_2, actor_critic_3, eval_envs_dic, env_name, num_processes,
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

        eval_attn_masks = (torch.sigmoid(actor_critic_0.base.linear_attention) > 0.5).float()
        eval_attn_masks1 = (torch.sigmoid(actor_critic_0.base.block1.attention) > 0.5).float()
        eval_attn_masks2 = (torch.sigmoid(actor_critic_0.base.block2.attention) > 0.5).float()
        eval_attn_masks3 = (torch.sigmoid(actor_critic_0.base.block3.attention) > 0.5).float()
    elif actor_critic_0.attention_size == 1:
        eval_attn_masks = torch.zeros(num_processes, actor_critic_0.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes,  16 , device=device)
        eval_attn_masks2 = torch.zeros(num_processes,  32 , device=device)
        eval_attn_masks3 = torch.zeros(num_processes,  32 , device=device)

    else:
        eval_attn_masks = torch.zeros(num_processes, *actor_critic_0.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes,  16 , device=device)
        eval_attn_masks2 = torch.zeros(num_processes,  32 , device=device)
        eval_attn_masks3 = torch.zeros(num_processes,  32 , device=device)


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

            max_policy = torch.max(torch.max(torch.max(dist_probs, dist_probs_1), dist_probs_2), dist_probs_3)
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
            if t==0:
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
            next_obs_sum = logger.obs_sum[env_name] + next_obs.cpu()
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

    rew_batch = np.array(rew_batch)
    int_rew_batch = np.array(int_rew_batch)
    done_batch = np.array(done_batch)
    seed_batch = np.array(seed_batch)
    # num_zero_obs_end = np.zeros_like(reward)
    # for i in range(len(reward)):
    #     if (obs_sum[i][0] == 0).sum() == 0:
    #         num_zero_obs_end[i]= 1

    return rew_batch, int_rew_batch, done_batch, seed_batch

def evaluate_procgen_ensemble(actor_critic, actor_critic_1, actor_critic_2, actor_critic_3, actor_critic_maxEnt, eval_envs_dic, env_name, num_processes,
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
    eval_recurrent_hidden_states = torch.zeros( num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    # eval_masks = torch.zeros(num_processes, 1, device=device)

    if attention_features:
        # eval_attn_masks = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        # eval_attn_masks1 = torch.zeros(num_processes, 16, device=device)
        # eval_attn_masks2 = torch.zeros(num_processes, 32, device=device)
        # eval_attn_masks3 = torch.zeros(num_processes, 32, device=device)

        eval_attn_masks  = (torch.sigmoid(actor_critic.base.linear_attention) > 0.5).float()
        eval_attn_masks1 = (torch.sigmoid(actor_critic.base.block1.attention) > 0.5).float()
        eval_attn_masks2 = (torch.sigmoid(actor_critic.base.block2.attention) > 0.5).float()
        eval_attn_masks3 = (torch.sigmoid(actor_critic.base.block3.attention) > 0.5).float()
    elif actor_critic.attention_size == 1:
        eval_attn_masks  = torch.zeros(num_processes, actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes,  16 , device=device)
        eval_attn_masks2 = torch.zeros(num_processes,  32 , device=device)
        eval_attn_masks3 = torch.zeros(num_processes,  32 , device=device)

    else:
        eval_attn_masks  = torch.zeros(num_processes, *actor_critic.attention_size, device=device)
        eval_attn_masks1 = torch.zeros(num_processes,  16 , device=device)
        eval_attn_masks2 = torch.zeros(num_processes,  32 , device=device)
        eval_attn_masks3 = torch.zeros(num_processes,  32 , device=device)

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
    is_novel = torch.ones(num_processes,1,dtype=torch.bool, device=device)
    m = FixedCategorical(torch.tensor([0.55, 0.25, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]).repeat(num_processes, 1))
    maxEnt_steps = torch.zeros(num_processes,1, device=device)
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

            dist_probs[:, 1] += dist_probs[:, 0]
            dist_probs[:, 1] += dist_probs[:, 2]
            dist_probs[:, 0] = 0
            dist_probs[:, 2] = 0

            dist_probs[:, 7] += dist_probs[:, 6]
            dist_probs[:, 7] += dist_probs[:, 8]
            dist_probs[:, 6] = 0
            dist_probs[:, 8] = 0
            pure_action = dist_probs.max(1)[1].unsqueeze(1)
            prob_pure_action = dist_probs.max(1)[0].unsqueeze(1)


            moving_average_prob = (1 - beta) * moving_average_prob + beta * prob_pure_action
            idex_prob = (prob_pure_action > 0.9)
            moving_average_prob = moving_average_prob*idex_prob

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

            dist_probs1[:, 1] += dist_probs1[:, 0]
            dist_probs1[:, 1] += dist_probs1[:, 2]
            dist_probs1[:, 0] = 0
            dist_probs1[:, 2] = 0

            dist_probs1[:, 7] += dist_probs1[:, 6]
            dist_probs1[:, 7] += dist_probs1[:, 8]
            dist_probs1[:, 6] = 0
            dist_probs1[:, 8] = 0
            pure_action1 = dist_probs1.max(1)[1].unsqueeze(1)
            prob_pure_action1 = dist_probs1.max(1)[0].unsqueeze(1)

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

            dist_probs2[:, 1] += dist_probs2[:, 0]
            dist_probs2[:, 1] += dist_probs2[:, 2]
            dist_probs2[:, 0] = 0
            dist_probs2[:, 2] = 0

            dist_probs2[:, 7] += dist_probs2[:, 6]
            dist_probs2[:, 7] += dist_probs2[:, 8]
            dist_probs2[:, 6] = 0
            dist_probs2[:, 8] = 0
            pure_action2 = dist_probs2.max(1)[1].unsqueeze(1)
            prob_pure_action2 = dist_probs2.max(1)[0].unsqueeze(1)

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

            dist_probs3[:, 1] += dist_probs3[:, 0]
            dist_probs3[:, 1] += dist_probs3[:, 2]
            dist_probs3[:, 0] = 0
            dist_probs3[:, 2] = 0

            dist_probs3[:, 7] += dist_probs3[:, 6]
            dist_probs3[:, 7] += dist_probs3[:, 8]
            dist_probs3[:, 6] = 0
            dist_probs3[:, 8] = 0
            pure_action3 = dist_probs3.max(1)[1].unsqueeze(1)
            prob_pure_action3 = dist_probs3.max(1)[0].unsqueeze(1)

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

            dist_probs_maxEnt[:, 1] += dist_probs_maxEnt[:, 0]
            dist_probs_maxEnt[:, 1] += dist_probs_maxEnt[:, 2]
            dist_probs_maxEnt[:, 0] = 0
            dist_probs_maxEnt[:, 2] = 0

            dist_probs_maxEnt[:, 7] += dist_probs_maxEnt[:, 6]
            dist_probs_maxEnt[:, 7] += dist_probs_maxEnt[:, 8]
            dist_probs_maxEnt[:, 6] = 0
            dist_probs_maxEnt[:, 8] = 0
            pure_action_maxEnt = dist_probs_maxEnt.max(1)[1].unsqueeze(1)
            prob_pure_action_maxEnt = dist_probs_maxEnt.max(1)[0].unsqueeze(1)


            # is_not_maxEnt = (pure_action == pure_action1) * (pure_action == pure_action2) * (prob_pure_action > 0.5) * (prob_pure_action1 > 0.5) * (prob_pure_action2 > 0.5)

            # env_steps = env_steps+1
            maxEnt_steps = maxEnt_steps - 1
            is_maxEnt_steps_limit = (maxEnt_steps<=0)
            is_equal = (action0 == action1) * (action0 == action2) * (action0 == action3)
            # step_count = (step_count+1)*is_equal
            # is_maxEnt = (step_count<10)
            # is_pure_action = is_novel*is_equal
            is_pure_action = is_equal*is_maxEnt_steps_limit
            maxEnt_steps = (m.sample() + 1).to(device)*is_pure_action + maxEnt_steps*(~is_pure_action)

            action = action0*is_pure_action + action_maxEnt*(~is_pure_action)
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
            next_obs, reward, done, infos = eval_envs.step(action0.squeeze().cpu().numpy())
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
            clean_reward = np.zeros_like(reward)
            clean_done = np.zeros_like(reward)
            for i in range(len(done)):
                seeds[i] = infos[i]['level_seed']
                if (t > 0 and np.array(done_batch)[:,i].sum() == 0):
                    clean_reward[i] = reward[i]
                    clean_done[i] = done[i]
            seed_batch.append(seeds)
            rew_batch.append(clean_reward)
            done_batch.append(clean_done)

            if t == 498:
                print("stop")

            is_novel = torch.zeros(num_processes, 1, dtype=torch.bool, device=device)
            for i in range(len(done)):
                if done[i] == 1:
                    logger.obs_sum[env_name][i] = next_obs[i].cpu()
                    is_novel[i] = True

            next_obs_sum = logger.obs_sum[env_name] + next_obs.cpu()
            for i in range(len(is_novel)):
                num_zero_obs_sum = (logger.obs_sum[env_name][i][0] == 0).sum()
                num_zero_next_obs_sum = (next_obs_sum[i][0] == 0).sum()
                if num_zero_next_obs_sum < num_zero_obs_sum:
                    is_novel[i] = True

            # for i, info in enumerate(infos):
            #     eval_episode_len_buffer[i] += 1
            #     if done[i] == True:
            #         eval_episode_rewards.append(reward[i])
            #         eval_episode_len.append(eval_episode_len_buffer[i])
            #         eval_episode_len_buffer[i] = 0

            logger.obs[env_name] = next_obs
            logger.obs_sum[env_name] = next_obs_sum

    rew_batch = np.array(rew_batch)
    done_batch = np.array(done_batch)
    seed_batch = np.array(seed_batch)
    # np.where(np.logical_xor(rew_batch, done_batch))
    # done_row_sum = done_batch.sum(0)
    # for i in range(len(done_row_sum)):
    #     if done_row_sum[i] == 0 :
    #         done_batch[steps-1,i] = True

    return rew_batch, done_batch
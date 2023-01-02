import numpy as np
import torch

from a2c_ppo_acktr import utils
# from a2c_ppo_acktr.envs import make_vec_envs
import matplotlib.pyplot as plt

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
            _, action, _, eval_recurrent_hidden_states, _, _, _, _ = actor_critic.act(
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
            _, action, _, eval_recurrent_hidden_states, _, _, _, _ = actor_critic.act(
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


def maxEnt_oracle(obs_all, action):
    next_action = torch.tensor(action)
    for i in range(len(action)):
        obs =  obs_all[i].cpu().numpy()
        action_i = action[i]


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
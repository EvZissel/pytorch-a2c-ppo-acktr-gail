import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, obs_rms, eval_envs_dic, eval_locations_dic ,env_name, seed, num_processes, num_tasks, eval_log_dir,
             device, deterministic=True, mid=False, **kwargs):

    eval_envs = eval_envs_dic[env_name]
    locations = eval_locations_dic[env_name]
    eval_episode_rewards = []
    num_uniform = 0

    for iter in range(0, num_tasks, num_processes):
        eval_actions = []
        for i in range(num_processes):
            eval_envs.set_task_id(task_id=iter+i, task_location=locations[i], indices=i)
        vec_norm = utils.get_vec_normalize(eval_envs)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.obs_rms = obs_rms

        obs = eval_envs.reset()
        eval_recurrent_hidden_states = torch.zeros(
            num_processes, actor_critic.recurrent_hidden_state_size, device=device)
        eval_masks = torch.zeros(num_processes, 1, device=device)
        eval_attn_masks = torch.zeros(num_processes, 8, device=device)
        if mid:
            eval_attn_masks = torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device)

        # while len(eval_episode_rewards) < 1:
        for t in range(kwargs["steps"]):
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states, eval_attn_masks = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    attn_masks=eval_attn_masks,
                    deterministic=deterministic)

            # Obser reward and next obs
            obs, _, done, infos = eval_envs.step(action.cpu())
            eval_actions.append(action)
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
        eval_actions = torch.stack(eval_actions).squeeze()
        for i in range(num_processes):
            if len(torch.unique(eval_actions[:, i])) == len(eval_actions[:, i]):
                # if torch.equal(torch.sort(eval_actions[:,i])[0], torch.tensor([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5], device=eval_actions.device)):
                num_uniform += 1


    # print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
    #     len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    return eval_episode_rewards, eval_actions, num_uniform

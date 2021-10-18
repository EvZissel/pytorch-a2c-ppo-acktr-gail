for i in {1..8}
do
nohup python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v8" --algo  ppo \
--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 \
--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 10 --eval-nondet_interval 10 \
--log-dir ppo_logs --task_steps=6  --obs_recurrent --val_agent_steps 1 --val_reinforce_update \
--no_normalize --free_exploration 6 --seed  $i --save-interval 1000 --gpu_device 1 > outputs/bandit_HA_out_25v8_v8_seed_"$i"_entropy_0.05_RandObsLocation.out &

sleep 3
done


for i in {1..8}
do
nohup python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v9" --algo  ppo \
--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 \
--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 10 --eval-nondet_interval 10 \
--log-dir ppo_logs --task_steps=6  --obs_recurrent --val_agent_steps 1 --val_reinforce_update \
--no_normalize --free_exploration 6 --seed  $i --save-interval 1000 --gpu_device 2 > outputs/bandit_HA_out_25v8_v9_seed_"$i"_entropy_0.05_RandObsLocation.out &

sleep 3
done

#nohup python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v8" --algo  ppo \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 15000000 --eval-interval 10 --eval-nondet_interval 10 \
#--log-dir ppo_logs --task_steps=6  --obs_recurrent --val_agent_steps 1 --hard_attn --val_reinforce_update \
#--no_normalize --free_exploration 6 --seed  0 --save-interval 1000 --gpu_device 2 --rotate > outputs/bandit_HA_out_25v8_v8_seed_0_entropy_0.05_rotate_noObs.out &
#
#nohup python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v9" --algo  ppo \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 15000000 --eval-interval 10 --eval-nondet_interval 10 \
#--log-dir ppo_logs --task_steps=6  --obs_recurrent --val_agent_steps 1 --hard_attn --val_reinforce_update \
#--no_normalize --free_exploration 6 --seed  0 --save-interval 1000 --gpu_device 3 --rotate > outputs/bandit_HA_out_25v8_v9_seed_0_entropy_0.05_rotate_noObs.out &

#for i in {1..10}
#do
#nohup python3 main.py --env-name "h_bandit-obs-randchoose-v8" --algo  ppo \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 10 --eval-nondet_interval 10 \
#--log-dir ppo_logs --task_steps=6  --obs_recurrent \
#--no_normalize --free_exploration 6 --seed  $i --save-interval 1000 --gpu_device 1 --rotate > outputs/bandit_out_25v8_seed_"$i"_entropy_0.05_RotAllRewardMag.out &
#
#sleep 3
#done

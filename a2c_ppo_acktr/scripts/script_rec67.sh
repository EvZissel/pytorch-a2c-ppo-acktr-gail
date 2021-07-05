python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6  --obs_recurrent --val_agent_steps 1 --hard_attn --val_reinforce_update --no_normalize --free_exploration 6 --seed  1 &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6  --obs_recurrent --val_agent_steps 1 --hard_attn --val_reinforce_update --no_normalize --free_exploration 6 --seed  2 &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6  --obs_recurrent --val_agent_steps 1 --hard_attn --val_reinforce_update --no_normalize --free_exploration 6 --seed  3 &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6  --obs_recurrent --val_agent_steps 1 --hard_attn --val_reinforce_update --no_normalize --free_exploration 6 --seed  4 &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6  --obs_recurrent --val_agent_steps 1 --hard_attn --val_reinforce_update --no_normalize --free_exploration 6 --seed  5 &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6  --obs_recurrent --val_agent_steps 1 --hard_attn --val_reinforce_update --no_normalize --free_exploration 6 --seed  6 &
sleep 3


wait

echo "obs recurrent 25 arms with 25 val=training arms, as a baseline, free exploration, dual_rl. Checking different hyperparamaters hard attention, now with REINFORCE updates"
echo "
Seed 6 Iter 3845 ten_arms 0.96 many_arms 0.14
Seed 3 Iter 3945 ten_arms 0.96 many_arms 0.15
Seed 4 Iter 3915 ten_arms 1.0 many_arms 0.17
Seed 1 Iter 3835 ten_arms 1.0 many_arms 0.1
Seed 2 Iter 3990 ten_arms 1.0 many_arms 0.14
Seed 5 Iter 3960 ten_arms 1.0 many_arms 0.15
"
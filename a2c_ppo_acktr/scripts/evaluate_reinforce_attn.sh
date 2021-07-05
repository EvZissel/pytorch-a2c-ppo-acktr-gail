for i in {1..10}
do
python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v8" --algo  ppo \
--log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 \
--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 \
--log-dir ./logs --task_steps=6  --obs_recurrent --val_agent_steps 1 --hard_attn --val_reinforce_update \
--no_normalize --free_exploration 6 --seed  $i &

sleep 3
done


for i in {1..10}
do
python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v9" --algo  ppo \
--log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 \
--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 \
--log-dir ./logs --task_steps=6  --obs_recurrent --val_agent_steps 1 --hard_attn --val_reinforce_update \
--no_normalize --free_exploration 6 --seed  $i &

sleep 3
done


wait


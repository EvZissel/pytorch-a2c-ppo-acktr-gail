python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 16000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_meanvargrad --meanvar_beta 1.25 --free_exploration 6 --seed  11 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 16000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_meanvargrad --meanvar_beta 1.25 --free_exploration 6 --seed  12 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 16000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_meanvargrad --meanvar_beta 1.25 --free_exploration 6 --seed  13 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 16000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_meanvargrad --meanvar_beta 1.25 --free_exploration 6 --seed  14 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 16000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_meanvargrad --meanvar_beta 1.25 --free_exploration 6 --seed  15 &
wait

echo "recurrent 25 arms and free exploration, meanvargrad beta 1.25"

python3 main.py --env-name "h_bandit-randchoose-v2" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad  --seed 1 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v2" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad  --seed 2 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v2" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad  --seed 3 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v2" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad  --seed 4 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v2" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad  --seed 5 &

sleep 3

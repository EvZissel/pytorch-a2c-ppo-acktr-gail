python3 main.py --env-name "h_bandit-randchoose-v6" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.03  --seed 1 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v6" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.03  --seed 2 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v6" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.03  --seed 3 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v6" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.03  --seed 4 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v6" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.03  --seed 5 &

wait

echo "check testgrad on 5 arms FF training, and noise 0.03, without free exploration, with the randomly generated domains, to compare with the baseline."
echo "noise 1.0 seemed to underfit, so adding less noise."
echo "Seed 2 Iter 5900 five_arms 17.32 ten_arms 14.14 many_arms 10.03 : this looks pretty well. Inspecting the plots shows that that test set performance did not yet plateau"
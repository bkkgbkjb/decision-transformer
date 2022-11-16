export CUDA_VISIBLE_DEVICES=1

nohup python experiment.py --env hopper --dataset medium --seed 0 --force-save-model True &>hm0.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium --seed 100 &>hm100.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium --seed 200 &>hm200.log &
sleep 2

nohup python experiment.py --env hopper --dataset medium-replay --seed 0 --force-save-model True &>hmr0.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium-replay --seed 100 &>hmr100.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium-replay --seed 200 &>hmr200.log &
sleep 2

nohup python experiment.py --env hopper --dataset medium-expert --seed 0 --force-save-model True &>hme0.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium-expert --seed 100 &>hme100.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium-expert --seed 200 &>hme200.log &
sleep 2


export CUDA_VISIBLE_DEVICES=2
nohup python experiment.py --env hopper --dataset medium --seed 0 --force-save-model True --feedback 3000 &>wm0.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium --seed 100 --feedback 3000 &>wm100.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium --seed 200 --feedback 3000 &>wm200.log &
sleep 2

nohup python experiment.py --env hopper --dataset medium-expert --seed 0 --force-save-model True --feedback 3000 &>wme0.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium-expert --seed 100 --feedback 3000 &>wme100.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium-expert --seed 200 --feedback 3000 &>wme200.log &
sleep 2

nohup python experiment.py --env hopper --dataset medium-replay --seed 0 --force-save-model True --feedback 3000 &>wmr0.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium-replay --seed 100 --feedback 3000 &>wmr100.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium-replay --seed 200 --feedback 3000 &>wmr200.log &
sleep 2


export CUDA_VISIBLE_DEVICES=3
nohup python experiment.py --env hopper --dataset medium --seed 0 --force-save-model True --feedback 500 &>cm0.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium --seed 100 --feedback 500 &>cm100.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium --seed 200 --feedback 500 &>cm200.log &
sleep 2

nohup python experiment.py --env hopper --dataset medium-expert --seed 0 --force-save-model True --feedback 500 &>cme0.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium-expert --seed 100 --feedback 500 &>cme100.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium-expert --seed 200 --feedback 500 &>cme200.log &
sleep 2

nohup python experiment.py --env hopper --dataset medium-replay --seed 0 --force-save-model True --feedback 500 &>cmr0.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium-replay --seed 100 --feedback 500 &>cmr100.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium-replay --seed 200 --feedback 500 &>cmr200.log &
sleep 2

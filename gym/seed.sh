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
nohup python experiment.py --env walker2d --dataset medium --seed 0 --force-save-model True &>wm0.log &
sleep 2
nohup python experiment.py --env walker2d --dataset medium --seed 100 &>wm100.log &
sleep 2
nohup python experiment.py --env walker2d --dataset medium --seed 200 &>wm200.log &
sleep 2

nohup python experiment.py --env walker2d --dataset medium-expert --seed 0 --force-save-model True &>wme0.log &
sleep 2
nohup python experiment.py --env walker2d --dataset medium-expert --seed 100 &>wme100.log &
sleep 2
nohup python experiment.py --env walker2d --dataset medium-expert --seed 200 &>wme200.log &
sleep 2

nohup python experiment.py --env walker2d --dataset medium-replay --seed 0 --force-save-model True &>wmr0.log &
sleep 2
nohup python experiment.py --env walker2d --dataset medium-replay --seed 100 &>wmr100.log &
sleep 2
nohup python experiment.py --env walker2d --dataset medium-replay --seed 200 &>wmr200.log &
sleep 2


export CUDA_VISIBLE_DEVICES=3
nohup python experiment.py --env halfcheetah --dataset medium --seed 0 --force-save-model True &>cm0.log &
sleep 2
nohup python experiment.py --env halfcheetah --dataset medium --seed 100 &>cm100.log &
sleep 2
nohup python experiment.py --env halfcheetah --dataset medium --seed 200 &>cm200.log &
sleep 2

nohup python experiment.py --env halfcheetah --dataset medium-expert --seed 0 --force-save-model True &>cme0.log &
sleep 2
nohup python experiment.py --env halfcheetah --dataset medium-expert --seed 100 &>cme100.log &
sleep 2
nohup python experiment.py --env halfcheetah --dataset medium-expert --seed 200 &>cme200.log &
sleep 2

nohup python experiment.py --env halfcheetah --dataset medium-replay --seed 0 --force-save-model True &>cmr0.log &
sleep 2
nohup python experiment.py --env halfcheetah --dataset medium-replay --seed 100 &>cmr100.log &
sleep 2
nohup python experiment.py --env halfcheetah --dataset medium-replay --seed 200 &>cmr200.log &
sleep 2

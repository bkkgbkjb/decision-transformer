export CUDA_VISIBLE_DEVICES=0

nohup python experiment.py --env hopper --dataset medium --seed 0 &>hm0.log &
nohup python experiment.py --env hopper --dataset medium --seed 100 &>hm100.log &
nohup python experiment.py --env hopper --dataset medium --seed 200 &>hm200.log &

nohup python experiment.py --env hopper --dataset medium-replay --seed 0 &>hmr0.log &
nohup python experiment.py --env hopper --dataset medium-replay --seed 100 &>hmr100.log &
nohup python experiment.py --env hopper --dataset medium-replay --seed 200 &>hmr200.log &

nohup python experiment.py --env hopper --dataset medium-expert --seed 0 &>hme0.log &
nohup python experiment.py --env hopper --dataset medium-expert --seed 100 &>hme100.log &
nohup python experiment.py --env hopper --dataset medium-expert --seed 200 &>hme200.log &

# nohup python experiment.py --env hopper --dataset random &>hr.log &

export CUDA_VISIBLE_DEVICES=1
nohup python experiment.py --env walker2d --dataset medium --seed 0 &>wm0.log &
nohup python experiment.py --env walker2d --dataset medium --seed 100 &>wm100.log &
nohup python experiment.py --env walker2d --dataset medium --seed 200 &>wm200.log &

nohup python experiment.py --env walker2d --dataset medium-expert --seed 0 &>wme0.log &
nohup python experiment.py --env walker2d --dataset medium-expert --seed 100 &>wme100.log &
nohup python experiment.py --env walker2d --dataset medium-expert --seed 200 &>wme200.log &

nohup python experiment.py --env walker2d --dataset medium-replay --seed 0 &>wmr0.log &
nohup python experiment.py --env walker2d --dataset medium-replay --seed 100 &>wmr100.log &
nohup python experiment.py --env walker2d --dataset medium-replay --seed 200 &>wmr200.log &

# nohup python experiment.py --env walker2d --dataset random &>wr.log &

export CUDA_VISIBLE_DEVICES=2
nohup python experiment.py --env halfcheetah --dataset medium --seed 0 &>cm0.log &
nohup python experiment.py --env halfcheetah --dataset medium --seed 100 &>cm100.log &
nohup python experiment.py --env halfcheetah --dataset medium --seed 200 &>cm200.log &

nohup python experiment.py --env halfcheetah --dataset medium-expert --seed 0 &>cme0.log &
nohup python experiment.py --env halfcheetah --dataset medium-expert --seed 100 &>cme100.log &
nohup python experiment.py --env halfcheetah --dataset medium-expert --seed 200 &>cme200.log &

nohup python experiment.py --env halfcheetah --dataset medium-replay --seed 0 &>cmr0.log &
nohup python experiment.py --env halfcheetah --dataset medium-replay --seed 100 &>cmr100.log &
nohup python experiment.py --env halfcheetah --dataset medium-replay --seed 200 &>cmr200.log &

# nohup python experiment.py --env halfcheetah --dataset random &>cr.log &
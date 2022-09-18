export CUDA_VISIBLE_DEVICES=5

nohup python experiment.py --env hopper --dataset medium &>hm.log &

nohup python experiment.py --env hopper --dataset medium-expert &>hme.log &

nohup python experiment.py --env hopper --dataset medium-replay &>hmr.log &
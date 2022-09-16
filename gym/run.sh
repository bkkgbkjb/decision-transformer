
nohup python experiment.py --env hopper --dataset medium --model_type dt &

nohup python experiment.py --env hopper --dataset medium-expert --model_type pdt --subepisode True &

nohup python experiment.py --env hopper --dataset expert --model_type dt
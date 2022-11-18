export CUDA_VISIBLE_DEVICES=1

nohup python experiment.py --env maze2d-umaze --dataset dense --seed 0 --force-save-model True &>hm0.log &
sleep 2
nohup python experiment.py --env maze2d-umaze --dataset dense --seed 100 &>hm100.log &
sleep 2
nohup python experiment.py --env maze2d-umaze --dataset dense --seed 200 &>hm200.log &
sleep 2

export CUDA_VISIBLE_DEVICES=2
nohup python experiment.py --env maze2d-medium --dataset dense --seed 0 --force-save-model True &>hmr0.log &
sleep 2
nohup python experiment.py --env maze2d-medium --dataset dense --seed 100 &>hmr100.log &
sleep 2
nohup python experiment.py --env maze2d-medium --dataset dense --seed 200 &>hmr200.log &
sleep 2

export CUDA_VISIBLE_DEVICES=3
nohup python experiment.py --env maze2d-large --dataset dense --seed 0 --force-save-model True &>hme0.log &
sleep 2
nohup python experiment.py --env maze2d-large --dataset dense --seed 100 &>hme100.log &
sleep 2
nohup python experiment.py --env maze2d-large --dataset dense --seed 200 &>hme200.log &
sleep 2
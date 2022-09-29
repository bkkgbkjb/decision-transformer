# OFFLINE PREFERENCE-GUIDED POLICY OPTIMIZATION

This is source code for training an OFFLINE PREFERENCE-GUIDED POLICY OPTIMIZATION(OPPO) agent as described in our manuscript.

## Commands

### Install
```shell
# in a virtual environment
pip install -r requirements.txt

# then install d4rl(https://github.com/Farama-Foundation/D4RL) following their instructions
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl


# then install datasets same as Decision Transformer
cd gym/data
python download_d4rl_datasets.py
```


### Training
```shell
cd gym
# in a virtualenv
python experiment.py -h
```

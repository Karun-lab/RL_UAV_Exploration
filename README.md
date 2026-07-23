# Documentation for developing DRL for autonomous exploration

## Setup docker
On the university server run
```bash
docker run -it --gpus all \
  --name rl-karun \
  --network host \
  -e DISPLAY=$DISPLAY \
  -e ACCEPT_EULA=Y \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
  -v ~/thesis:/workspace/isaaclab/rl_WorkSpace \
  nvcr.io/nvidia/isaac-lab:2.1.0 bash
```

1. **Start docker**
```bash
docker start rl-karun
```
2. **To open docker**
```bash
docker exec -it rl-karun bash
```
Inside the docker

3. **Export the following to bashrc to let Isaac-lab know where Isaac-sim is**

```bash
export ISAACSIM_PATH=/isaac-sim
export ISAACSIM_PYTHON=/isaac-sim/python.sh
export ISAACSIM_ROOT=/isaac-sim00
```
4. **Clone repo**
```bash
cd /workspace/isaaclab
```
```bash
git clone https://github.com/Karun-lab/RL_UAV_Exploration.git
```

5. **Change folder**
```bash
cp -a RL_UAV_Exploration/. rl_WorkSpace/
```

## Folder, file  & code - its purpose
>agent folder contains SKRL config files (hyper-parameters)

>models folder contains USD files of drone and offices

>rl_envs folder contains main codes

>scripts folder contains code to start training and playing code from rl_envs

>__init__.py contains initialization and registers 

## Main ICM code
>iris_icm_exploration.py : It containes the RL environment for training, observation space, action space, reward fucntion

>iri_icm_agent.py: It contains hyper-paramters fo ICM exploration training

## Training
Recommeded to use tmux for training

```bash
cd /workspace/isaaclab
```
```bash
tmux new -s rl
```
```bash
CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh \
    rl_WorkSpace/scripts/train_iris_icm.py \
    --num_envs 16 --headless --enable_cameras
```
To exit tmux press Ctrl+b then d

## Play
Once training is complete
```bash
cd /workspace/isaaclab
```
```bash
CUDA_VISIBLE_DEVICES=1 /isaac-sim/python.sh \
  rl_WorkSpace/scripts/play_iris_icm.py \
  --task Isaac-Iris-ICM-v1 \
  --num_envs 1 \
  --headless \
  --livestream 2 \
  --enable_cameras
```

## ICM + VIO code
>iris_icm_map_env.py : It containes the RL environment for training, observation space, action space, reward fucntion. This uses drone's pose to create spatial memory

>iri_icm_map_agent.py: It contains hyper-paramters fo ICM exploration training

## Training
Recommeded to use tmux for training

```bash
cd /workspace/isaaclab
```
```bash
tmux new -s rl
```
```bash
CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh \
    rl_WorkSpace/scripts/train_icm_map.py \
    --num_envs 16 --headless --enable_cameras
```
To exit tmux press Ctrl+b then d

## ICM + LSTM code
>iris_icm_lstm_env.py : It containes the RL environment for training, observation space, action space, reward fucntion. This uses RNN to have LSTM

>iri_icm_lstm_agent.py: It contains hyper-paramters fo ICM exploration training

## Training
Recommeded to use tmux for training

```bash
cd /workspace/isaaclab
```
```bash
tmux new -s rl
```
```bash
CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh \
    rl_WorkSpace/scripts/train_icm_lstm.py \
    --num_envs 16 --headless --enable_cameras
```
To exit tmux press Ctrl+b then d
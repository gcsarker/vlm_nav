<div align="center">
<h1>VLM-NAV</h1>
</div>

<p> This work develops a UAV navigation and obstacle avoidance system based on monocular vision using a vision-language model (Google Gemini flash model) using the AirSim Simulator. RGB images captured by the front camera are processed into depth maps using the Depth-Anything algorithm. The generated depth maps are then translated into three discrete actions. </p>

## Observation Space
| No | State |
|:-|-:|
| 1 | RGB stream|
| 2 | Left distance sensor  |
| 3 | Right distance Sensor |
| 4 | Relative Heading Angle |


## Action Space
Our system works with three discrete action space.

| No | Name | Action |
|:-|-:|:-:|
| 1 | Forward | Move Straight |
| 2 | Left | Yaw left 25 degree/s |
| 3 | Right | Yaw Right 25 degree/s |

### Prepraration

1. **Build AirSim**: Follow the steps mentioned in <u>[AirSim Build Guide](https://microsoft.github.io/AirSim/build_windows/)</u>. Then prepare your custom Unreal Engine environment with Airsim Plugin. Refer to the tutorial for <u>[creating and setting up custom environment](https://microsoft.github.io/AirSim/unreal_custenv/)</u>.

2. **Setup AirSim Settings**: Copy the settings.json file from the AirSim_Settings folder to "Documents/AirSim/" in your local machine.

3. **Install Pytorch**: Please follow the steps in the pytorch install guide on their <u>[official website](https://pytorch.org/get-started/locally/)</u>.] or use the following pip command. 

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

4. **Install VLM-Nav**
```bash
git clone https://github.com/gcsarker/vlm_nav
cd vlm_nav
pip install -r requirements.txt
```

5. **Create API Key**: Create a  Google Developer API from <u>[Goole AI Studio](https://aistudio.google.com/)</u> and copy paste the API key to the config.ini file. Here are the details regarding different settings in the config file.

6. **Modify Config File**:
```
# Place your API Key Here
API_KEY = Enter_YOUR_API_KEY_HERE 
API_REQUEST_DELAY = 0.1

# Threshold values
TAU = 8
TAU_D = 4

# Desired velocity, height and yaw angle
VELOCITY = 3
FLIGHT_HEIGHT = 8
YAW_ANGLE = 25

# True for Logging flight details (images, position, angle)
FLIGHT_LOG = True

# dimensions of the map in x and y axis
MAP_DIM_X = 20
MAP_DIM_Y = 20

# True for estimating depth from the scene image
# if false, depthmap will be generated by AirSim API
DEPTH_ESTIMATION = False

# Duration for forward motion
FORWARD_DURATION = 1

# Duration in a open space
OPEN_SPACE_FORWARD_DURATION = 5

```

### Use our models
```python
from vlm_nav import VLMNav

# navigationMode
# 'flyAround' : flying using keyboard control 
# 'FCN' : Autonomous flying using trained navigator

nav = VLMNav(
    n_episodes = 10,
    spawn_coord= [0, 10],   # Specify Spawn Coordinate on the map
    target_coord = [100, 20], # Specify Target Coordinate on the map
    navigationMode='flyAround'
    )
nav.navigate()
```
To train the navigation model use the following format.

```python
from vlm_nav import navLearner
learner = navLearner(
    collect_steps = True, 
    n_steps = 10000, 
    batch_size = 32, 
    learning_rate = 0.001, 
    n_epochs = 100
    )
learner.learn()
```


## Citation

If you find this project useful, please consider citing:

```bibtex
@article{vlm_nav,

}

```

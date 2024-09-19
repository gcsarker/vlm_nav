import numpy as np
import csv
import requests
from tqdm import tqdm
import airsim
import math
import random
import os
import time
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

from vlm_nav.configurations import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# depth_anything_checkpoint = f'checkpoints/depth_anything_v2_{encoder}.pth'
# # To download DepthAnythingV2 checkpoint
# import requests
# if not os.path.exists(depth_anything_checkpoint):
#     response = requests.get()
#     with open(depth_anything_checkpoint, 'wb') as f:
#         f.write(response.content)

minimum_flight_height = flight_height+2
maximum_flight_height = flight_height-2
scene_dim = (144,256)
d_height, d_width = (144, 256)
angle_multiplier = 1

class droneEnvironment:
    def __init__(self):
        # OBSERVATION SPACE
        # front_camera : RGB feed from front camera
        # left_distance_sensor : distance sensor to the left
        # right_distance_sensor : distance sensor to the right
        # angle : Relative Angle between drone's forward direction and target location
        self.states = {
            'front_camera': np.zeros([d_height, d_width], dtype=float),
            'left_distance_sensor': 0.0,
            'right_distance_sensor': 0.0,
            'angle': 0.0,
        }

        # ACTION SPACE
        # 0 : Move Forward
        # 1 : Yaw left
        # 2 : Yaw right
        self.action_space = [0, 1, 2]

        self.drone = airsim.MultirotorClient()
        self.drone.confirmConnection()

        # Download the checkpoint file and load the model
        if depth_estimation:
            depth_anything_checkpoint = "checkpoints/depth_anything_v2_vits.pth"
            checkpoint_url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true"
            if not os.path.exists(depth_anything_checkpoint):
                self.download_checkpoint(
                    checkpoint_url,
                    depth_anything_checkpoint
                )
            else:
                print(f"{depth_anything_checkpoint} already exists. Skipping download.")

            self.transform = Compose([
                Resize(
                    width=252,
                    height=140,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
            self.depth_anything_v2 = DepthAnythingV2(encoder= 'vits', features= 64, out_channels= [48, 96, 192, 384])
            self.depth_anything_v2.load_state_dict(torch.load(depth_anything_checkpoint, map_location=DEVICE))
            self.depth_anything_v2 = self.depth_anything_v2.to(DEVICE).eval()


        # Scene Image requests to airsim for the front camera
        self.sceneReq = airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        
        # Depth Image request to airsim for the front camera
        self.depthReq = airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
        
        self.episode_counter = 0
        

    def download_checkpoint(self, url, file_path):
        # Get the file size from the headers
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))  # Total size in bytes

        # Progress bar
        with open(file_path, 'wb') as f, tqdm(
            desc=file_path,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                bar.update(len(data))

    def compute_distance(self, current_position, target_position):
        delta_x = (target_position[0] - current_position[0])**2
        delta_y = (target_position[1] - current_position[1])**2
        return (delta_x + delta_y)**(1/2)
    
    def compute_relative_angle(self, current_pos, target_pos, yaw, degree = False):
        angle_to_target = math.atan2(target_pos[1] - current_pos[1], target_pos[0]- current_pos[0])
        relative_angle = angle_to_target - yaw
        while relative_angle >= math.pi:
            relative_angle -= 2*math.pi
        while relative_angle < -math.pi:
            relative_angle += 2*math.pi

        if degree:
            relative_angle = math.degrees(relative_angle)
    
        return relative_angle
    
    # Function to write data to CSV at each timestep
    def write_to_csv(self, file_name, data):
        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def reset(self, spawn_coord, target_coord):
        self.step_counter = 0 # step counting for each episode
        self.episode_counter +=1
        self.collision = False 

        if flight_path_logging:
            self.episode_path = 'vlm_nav/flight_log/episode_'+str(self.episode_counter)
            os.makedirs(os.path.join(self.episode_path, 'scene'), exist_ok=True)
            os.makedirs(os.path.join(self.episode_path, 'depth'), exist_ok=True)
            with open(self.episode_path + '/flight_log.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'position',
                    'target',
                    'left_distance',
                    'right_distance',
                    'angle',
                    'collided'
                ])
        
        if spawn_coord is None:
            # Selecting random spawn loction
            spawn_side = np.random.choice([-1, 1]) # -1 for left, 1 for right side
            self.spawn_location = np.round(
                    np.array((
                        0.0,
                        spawn_side*random.uniform(0, map_y)
                        ), dtype = float),
                    decimals=2
                    )
        else:
            self.spawn_location = np.array(spawn_coord, dtype =float)
        
        if target_coord is None:
            # selecting random target location
            target_side = np.random.choice([-1, 1]) # -1 for left, 1 for right side
            self.target_location = np.round(
                    np.array((
                        map_x,
                        target_side*random.uniform(0, map_y)
                        ), dtype = float),
                    decimals=2
                    )
        else:
            self.target_location = np.array(target_coord, dtype = float)
        
        # Target Location Marker
        target_marker_pose = airsim.Pose(
            position_val=airsim.Vector3r(x_val=self.target_location[0], y_val=self.target_location[1], z_val=0),
            orientation_val=None
            )
        self.drone.simSetObjectPose('locMarker', target_marker_pose)

        self._setup_flight()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info 

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # moving to z value then spawn location otherwise problem occurs
        self.drone.moveToZAsync(z= flight_height, velocity=velocity).join()
        pos = airsim.Vector3r(x_val= self.spawn_location[0], y_val= self.spawn_location[1], z_val= flight_height)
        orient = airsim.Quaternionr(w_val= 1.0, x_val= 0.0, y_val= 0.0, z_val= 0)
        pose = airsim.Pose(position_val=pos, orientation_val=orient)
        self.drone.simSetVehiclePose(pose,ignore_collision=True)

    # To estimate depthmap from DepthAnything
    def estimate_depthmap(self, scene):
        scene = scene/255
        scene = self.transform({'image': scene})['image']
        scene = torch.from_numpy(scene).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            depth = self.depth_anything_v2(scene)
        depth = F.interpolate(depth[None], (d_height, d_width), mode='bilinear', align_corners=False)[0, 0]
        depth = depth.cpu().numpy() # converting to numpy array
        return depth

    # To get depthmap from airsim
    def get_depthmap(self, depthResponse):
        depth = np.array(depthResponse.image_data_float, dtype=float)
        depth = 255 / np.maximum(np.ones(depth.size), depth)
        depth = np.reshape(depth, (depthResponse.height, depthResponse.width))
        return depth

    def _get_obs(self):
        if self.step_counter == 0:
            pos = airsim.Vector3r(x_val= self.spawn_location[0], y_val= self.spawn_location[1], z_val= flight_height)
            orient = airsim.Quaternionr(w_val= 1.0, x_val= 0.0, y_val= 0.0, z_val= 0)
            pose = airsim.Pose(position_val=pos, orientation_val=orient)
        else:
            pose = self.drone.simGetVehiclePose()

        # Position          add /map_size        
        self.position = np.array((pose.position.x_val, pose.position.y_val), dtype=float)

        # distance to target
        self.distance = self.compute_distance(self.position, self.target_location)

        if depth_estimation:
            # Taking Scene image from airsim for depth estimation
            # This returns tensor depthmap, only converted to numpy array for saving
            scene_response = self.drone.simGetImages([self.sceneReq])[0]
            scene = np.frombuffer(scene_response.image_data_uint8, dtype=np.uint8)
            scene = scene.reshape(scene_dim[0], scene_dim[1], 3)
            depth = self.estimate_depthmap(scene)
            # depth = self.model.infer_image(scene)
        else:
            # Taking depthmap directly from airsim
            responses = self.drone.simGetImages([self.sceneReq, self.depthReq])
            scene = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            scene = scene.reshape(scene_dim[0], scene_dim[1], 3)
            depth = self.get_depthmap(responses[1])
        
        self.states['front_camera'] = depth 

        # distance sensors measurement
        left_distance = self.drone.getDistanceSensorData(distance_sensor_name = "left_distance_sensor").distance
        right_distance = self.drone.getDistanceSensorData(distance_sensor_name = "right_distance_sensor").distance
        self.states['left_distance_sensor'] = left_distance
        self.states['right_distance_sensor'] = right_distance

        # relative angle to the target
        _, _, yaw  = airsim.utils.to_eularian_angles(pose.orientation) 
        angle = self.compute_relative_angle(self.position, self.target_location, yaw)
        self.states['angle'] = angle*angle_multiplier

        
        # Logging
        if flight_path_logging:
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = depth*255
            depth = 255- depth
            cv2.imwrite(self.episode_path+'/depth/'+str(self.step_counter)+'.jpg', depth) # saving depth
            cv2.imwrite(self.episode_path+'/scene/'+str(self.step_counter)+'.jpg', scene) # saving scene
            
            self.write_to_csv(self.episode_path + '/flight_log.csv', [
                        self.position,
                        self.target_location,
                        left_distance,
                        right_distance,
                        angle,
                        self.collision
                    ])
        return self.states
    
    def _get_info(self):
        
        return {
            'position' : self.position,
            'spawn_location' : self.spawn_location,
            'target_location': self.target_location,
            'left_distance' : self.states['left_distance_sensor'],
            'right_distance': self.states['right_distance_sensor'],
            'angle': math.degrees(self.states['angle']),
        }

    def step(self, action, forward_duration = 2):
        if self.distance < 8:
            forward_duration = 1
        
        position = self.drone.simGetVehiclePose().position
        orientation = self.drone.simGetVehiclePose().orientation
        flight_height = position.z_val

        # keeping the position within range
        if (position.z_val < maximum_flight_height) or (position.z_val > minimum_flight_height):
            self.drone.moveToZAsync(z = flight_height, velocity=velocity).join()

        
        _, _, yaw = airsim.utils.to_eularian_angles(orientation)

        vx = math.cos(yaw) * velocity
        vy = math.sin(yaw) * velocity

        # Move Forward
        if action == 0:
            self.drone.moveByVelocityZAsync(vx=vx,
                                            vy=vy,
                                            z= flight_height,
                                            duration= forward_duration,
                                            drivetrain=airsim.DrivetrainType.ForwardOnly).join() 
            
        # Yaw left
        if action == 1:
            self.drone.rotateByYawRateAsync(yaw_rate= -yaw_angle, duration=1).join() 

        # Yaw right
        if action == 2:
            self.drone.rotateByYawRateAsync(yaw_rate= yaw_angle, duration=1).join()

        self.step_counter += 1
        obs = self._get_obs()
        info = self._get_info()
        
        terminated = False
        self.collision = False

        reward = 0.0
        if self.distance < 3:
            reward = 1.0
            self.drone.moveToZAsync(z=0, velocity=velocity).join()
            self.drone.landAsync().join()
            terminated = True
        
        if self.drone.simGetCollisionInfo().has_collided:
            print('Drone Collided!!!')
            self.collision = True
            terminated = True
            


        return obs, reward, terminated, False, info

    def render(self):
        pass
    
    def close(self):
        pass
        

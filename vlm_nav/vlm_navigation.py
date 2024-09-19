import sys
import numpy as np
import math
import keyboard
import torch
from vlm_nav.playground import droneEnvironment
from vlm_nav.navigator import Navigator
from vlm_nav.vlm import VLM
from vlm_nav.configurations import *
print('Current Configurations:')
print_config()

angle_tolerance = math.radians(yaw_angle)

# Navigation Mode:
# 'ruleBased' = Navigate following certain rules placed on the observation
# 'FCN' = Navigate by using a Fully Connected Network (FCN) that generate action
# 'FlyAround' = Navigate by manual steering

class VLMNav:
    def __init__(self, n_episodes = 10, spawn_coord = None, target_coord = None, navigationMode = 'ruleBased'):
        super(VLMNav, self).__init__()
        self.spawn_coord = spawn_coord
        self.target_coord = target_coord
        self.n_episodes = n_episodes
        # self.target_location = target_location
        self.navigationMode = navigationMode
        self.env = droneEnvironment()
        self.vlm = VLM()
        if self.navigationMode == 'FCN':
            self.navigator = Navigator()
            self.navigator.load_model()
        elif self.navigationMode =='flyAround':
            print("Please Select Action ('w': forward, 'a': yaw left, 'd': yaw right). Press q to quit!!")
    

    # compute mean of an image patch
    def compute_image_patch_mean(self, depthmap, patch_shape = [40, 50]):
        h, w = depthmap.shape[-2], depthmap.shape[-1]  # Get height and width of the tensor
        patch_height, patch_width = patch_shape
        # patch midpoints left to right
        midPoint1 = patch_width//2
        midpoint2 = w//3 
        midpoint3 = w//2 
        midpoint4 = midpoint2+ midpoint2 
        midpoint5 = w-patch_width//2
        midpoints = [midPoint1, midpoint2, midpoint3, midpoint4, midpoint5]
        
        rectMeans = []
        for midpoint in midpoints:
            start_y = (h - patch_height) // 2
            start_x = midpoint - (patch_width//2)
            rect = depthmap[..., start_y:start_y + patch_height, start_x:start_x + patch_width]
            rectMean = np.mean(rect)
            rectMeans.append(rectMean)
            
        return rectMeans

    def navigate(self):
        for i in range(self.n_episodes):
            observation, info = self.env.reset(self.spawn_coord, self.target_coord)            
            terminated = False
            prev_action = 0
            while not terminated:
                # to add delay to gemini API request (it can be ignored)
                # time.sleep(api_request_delay)
                
                depth = observation['front_camera'] 
                left_distance = observation['left_distance_sensor']
                right_distance = observation['right_distance_sensor']
                angle = observation['angle']
                angle = np.round(angle, 2)

                #mean of patches from left to right
                p1Mean, p2Mean, p3Mean, p4Mean, p5Mean = self.compute_image_patch_mean(depth)

                lObjDetected = left_distance < tau_d
                p1Detected = p1Mean > tau
                fObjDetected = (p2Mean > tau) or (p3Mean > tau) or (p4Mean >tau)
                p5Detected = p5Mean > tau
                rObjDetected = right_distance < tau_d
                
                # debug info
                print(f"ldist: {lObjDetected}, 1: {p1Mean : .2f}, 2: {p2Mean : .2f}, 3: {p3Mean : .2f},"
                      f"4: {p4Mean:.2f}, 5: {p5Mean :.2f}, rdist: {rObjDetected} angle: {angle}")

                # Manual Rule Based Nav
                if self.navigationMode == 'ruleBased':
                    action = self.selectActionRuleBased(depth, angle, lObjDetected, p1Detected, fObjDetected, p5Detected,
                                                    rObjDetected, prev_action)
                elif self.navigationMode == 'flyAround':
                    action = self.selectActionManual()
                
                elif self.navigationMode == 'FCN':
                    action = self.selectActionFCN(depth, angle, lObjDetected, p1Detected, fObjDetected, p5Detected, 
                                                  rObjDetected, prev_action)
                
                # To move the drone forward for large duration when no obstacle detected (helpful for big environment)
                if (p1Mean < 4) and (p2Mean < 4) and (p3Mean< 4) and (p4Mean< 4) and (p5Mean< 4):
                        fwdDuration = open_area_forward_duration
                else:
                        fwdDuration = forward_duration

                next_observation, reward, terminated, _ , info = self.env.step(action, fwdDuration)
                prev_action = action
                observation = next_observation

    def selectActionRuleBased(self, depth, angle, lObjDetected, p1Detected, fObjDetected, p5Detected, rObjDetected, prev_action):
        if (not lObjDetected) and (not p1Detected) and (not fObjDetected) and (not p5Detected) and (not rObjDetected):
            if abs(angle) > angle_tolerance:
                action = 1 if angle < 0 else 2
            else:
                action = 0
        
        elif fObjDetected:
            
            direction = self.vlm.get_vlm_feedback(depth)
            
            if direction['left'] and (not direction['right']):
                action = 1
            elif direction['right'] and (not direction['left']):
                action = 2
            else:
                action = 1 if angle < 0 else 2

        else: # either left or right side detected object
            
            if ((angle < 0) and lObjDetected) or ((angle < 0) and p1Detected) or ((angle > 0) and rObjDetected) or ((angle > 0) and p5Detected):
                action = 0 # go forward
            else:
                if abs(angle) > angle_tolerance:
                    action = 1 if angle < 0 else 2
                else:
                    action = 0

        # to prevent repetitive left right motion
        if (prev_action == 2 and action == 1) or (prev_action == 1 and action == 2):
                action = 0

        return action

    def selectActionFCN(self, depth, angle, lObjDetected, p1Detected, fObjDetected, p5Detected, rObjDetected, prev_action):
        isAnglePositive = 0 if angle < 0 else 1
        isAngleOverLimit = 0 if abs(angle) < angle_tolerance else 1
        prev_action_0 = 1 if prev_action == 0 else 0
        prev_action_1 = 1 if prev_action == 1 else 0
        prev_action_2 = 1 if prev_action == 2 else 0
        
        vlm_feedback_none = 0
        vlm_feedback_left = 0
        vlm_feedback_right = 0
        vlm_feedback_both = 0

        if not fObjDetected:
            vlm_feedback_none = 1 # it doesnot matter which value
        else:
            direction = self.vlm.get_vlm_feedback(depth)
            if direction['left'] and (not direction['right']):
                vlm_feedback_left = 1
            elif direction['right'] and (not direction['left']):
                vlm_feedback_right = 1
            else:
                vlm_feedback_both = 1
               
        inp = torch.tensor([isAnglePositive, isAngleOverLimit, lObjDetected, p1Detected, fObjDetected, p5Detected, rObjDetected,
               vlm_feedback_none, vlm_feedback_left, vlm_feedback_right, vlm_feedback_both, 
               prev_action_0, prev_action_1, prev_action_2], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = torch.softmax(self.navigator(inp), dim=1)
            _, action = torch.max(pred.data, 1)

        return action.item()
   
    def selectActionManual(self):
        while True:
            key_event = keyboard.read_event(suppress=True)
            if (key_event.event_type == keyboard.KEY_DOWN) and (key_event.name =='w'):
                action = 0
                print('selected action : Forward')
                break
            elif (key_event.event_type == keyboard.KEY_DOWN) and (key_event.name =='a'):
                action = 1
                print('selected action : Yaw Left')
                break
            elif (key_event.event_type == keyboard.KEY_DOWN) and (key_event.name =='d'):
                action = 2
                print('selected action : Yaw Right')
                break
            elif (key_event.event_type == keyboard.KEY_DOWN) and (key_event.name =='q'):
                sys.exit()
            else:
                continue
        
        return action
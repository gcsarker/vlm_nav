import csv
import keyboard
import numpy as np
import pandas as pd
import math
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from vlm_nav.configurations import *
from vlm_nav.playground import droneEnvironment
from vlm_nav.navigator import Navigator
from vlm_nav.vlm import VLM
angle_tolerance = math.radians(yaw_angle)

class navLearner():
    def __init__(self, 
                 collect_steps = True, 
                 n_steps = 10000, 
                 batch_size = 32, 
                 learning_rate = 0.001, 
                 n_epochs = 100) -> None:
        
        self.env = droneEnvironment()
        self.vlm = VLM()
        self.navigator = Navigator()
        self.dataset_path = 'vlm_nav/dataset/train_data.csv'
        self.collect_steps = collect_steps
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.lr = learning_rate
        self.n_epochs = n_epochs

    
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

    # Function to write data to CSV at each timestep
    def write_to_csv(self, file_name, data):
        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def collect_trajectories(self):
            
            with open(self.dataset_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'angle',
                    'lObjDetected', 
                    'p1Detected',
                    'fObjDetected', 
                    'p5Detected',
                    'rObjDetected',
                    'vlm_feedback',  
                    'prev_action',
                    'action'
                ])

            step = 1
            while step < self.n_steps:
                observation, info = self.env.reset()
                terminated = False
                prev_action = 0
                while not terminated:                    
                    # to add delay to gemini API request
                    # time.sleep(0.2)
                    
                    depth = observation['front_camera'] 
                    left_distance = observation['left_distance_sensor']
                    right_distance = observation['right_distance_sensor']
                    angle = observation['angle']
                    angle = np.round(angle, 2)

                    #mean of patches from left to right
                    p1Mean, p2Mean, p3Mean, p4Mean, p5Mean = self.compute_image_patch_mean(depth)
                    lObjDetected = left_distance < tau_d
                    fObjDetected = (p2Mean > tau) or (p3Mean > tau) or (p4Mean >tau)
                    rObjDetected = right_distance < tau_d

                    p1Detected = p1Mean > tau
                    p5Detected = p5Mean > tau
                    
                    if (not lObjDetected) and (not fObjDetected) and (not rObjDetected):
                        vlm_feedback = 'none' 
                    else:
                        direction = self.vlm.get_vlm_feedback(depth)
                        if direction['left'] and (not direction['right']):
                            vlm_feedback = 'left' # go left
                        elif direction['right'] and (not direction['left']):
                            vlm_feedback = 'right' # go right
                        elif direction['right'] and direction['left']:
                            vlm_feedback = 'both' # Select based on Angle
                        else:
                            vlm_feedback = 'center' # go forward
                    
                    print(f"p1: {p1Detected}, f: {fObjDetected}, p5: {p5Detected}, ldist: {lObjDetected}, rdist: {rObjDetected}, vlm: {vlm_feedback}")
                    
                    # Select Action
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
                        else:
                            continue

                    self.write_to_csv(self.dataset_path, [
                        angle,
                        lObjDetected, 
                        p1Detected,
                        fObjDetected,
                        p5Detected, 
                        rObjDetected, 
                        vlm_feedback,  
                        prev_action,
                        action
                    ])
                    
                    next_observation, reward, terminated, _ , info = self.env.step(action)
                    observation = next_observation
                    prev_action = action

                    step += 1
                    if step >= self.n_steps:
                        break
    

    def load_data(self):
        df = pd.read_csv(self.dataset_path)
        df['isAnglePositive'] = np.where(df['angle'] < 0, 0, 1)
        df['isAngleOverLimit'] = np.where(np.abs(df['angle']) < angle_tolerance, 0, 1)
        df['vlm_feedback_none'] = np.where(df['vlm_feedback'] == 'none', 1, 0)
        df['vlm_feedback_left'] = np.where(df['vlm_feedback'] == 'left', 1, 0)
        df['vlm_feedback_right'] = np.where(df['vlm_feedback'] == 'right', 1, 0)
        df['vlm_feedback_both'] = np.where(df['vlm_feedback'] == 'both', 1, 0)
        df['prev_action_0'] = np.where(df['prev_action'] == 0, 1, 0)
        df['prev_action_1'] = np.where(df['prev_action'] == 1, 1, 0)
        df['prev_action_2'] = np.where(df['prev_action'] == 2, 1, 0)
        df['action_0'] = np.where(df['action'] == 0, 1, 0)
        df['action_1'] = np.where(df['action'] == 1, 1, 0)
        df['action_2'] = np.where(df['action'] == 2, 1, 0)
        
        x_columns = ['isAnglePositive', 'isAngleOverLimit', 'lObjDetected', 'p1Detected', 'fObjDetected', 'p5Detected', 'rObjDetected', 
                      'vlm_feedback_none', 'vlm_feedback_left', 'vlm_feedback_right', 'vlm_feedback_both', 
                      'prev_action_0', 'prev_action_1', 'prev_action_2']

        y_columns = ['action_0','action_1', 'action_2']
        
        X = df[x_columns].to_numpy(dtype=float)
        X = torch.tensor(X, dtype=torch.float32)
        y = df[y_columns].to_numpy(dtype=float)
        y = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def learn(self):
        print('Collecting flight trajectories for trainng...')
        if self.collect_steps:
            self.collect_trajectories()
        print('Flight trajectories collection complete')

        print('Loading Data for model training...')
        train_loader, val_loader = self.load_data()
        
        optimizer = optim.Adam(self.navigator.parameters(), lr=self.lr)
        loss = nn.CrossEntropyLoss()

        print('training navigation model...')
        best_loss = 100
        for epoch in range(self.n_epochs):
            self.navigator.train()
            train_loss = 0
            train_acc = 0
            for X, y in train_loader:
                optimizer.zero_grad()
                pred = self.navigator(X)
                loss_value = loss(pred, y)
                loss_value.backward()
                optimizer.step()

                #Accuracy
                pred = torch.softmax(pred, dim=1)
                _, predicted = torch.max(pred.data, 1)
                _, actual = torch.max(y.data, 1)
                correct = (predicted == actual).sum().item()
                acc_value = 100 * correct / self.batch_size

                train_loss += loss_value.item()
                train_acc += acc_value

            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = train_acc / len(train_loader)

            self.navigator.eval()
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():  # Disable gradient computation for validation
                for val_data, val_labels in val_loader:
                    val_pred = self.navigator(val_data)
                    loss_value = loss(val_pred, val_labels)
                    val_loss += loss_value.item()

                    # Compute accuracy
                    val_pred = torch.softmax(val_pred, dim=1)
                    _, predicted = torch.max(val_pred.data, 1)
                    _, actual = torch.max(val_labels.data, 1)
                    correct = (predicted == actual).sum().item()
                    acc_value = 100 * correct / self.batch_size
                    val_acc += acc_value

            # Compute average validation loss
            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_acc/ len(val_loader)

            print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.2f}, Training Accuracy: {avg_train_acc:.2f},"
                    f"validation Loss: {avg_val_loss:.2f}, validation Accuracy: {avg_val_acc:.2f},")
            
            if avg_val_loss < best_loss:
                print('Saving Best Model...')
                self.navigator.save_model()
                best_loss = avg_val_loss

            print('Training complete...')

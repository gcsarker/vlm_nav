import torch
import torch.nn as nn

# Variables to be passed to the action model
# 'isAnglePositive'
# 'isAngleOverLimit' 
# 'lObjDetected' 
# 'p1Detected' 
# 'fObjDetected' 
# 'p5Detected' 
# 'rObjDetected' 
# 'vlm_feedback_none' 
# 'vlm_feedback_left' 
# 'vlm_feedback_right' 
# 'vlm_feedback_both', 
# 'prev_action_0', 
# 'prev_action_1', 
# 'prev_action_2'

class Navigator(nn.Module):
    def __init__(self, input_size = 14, 
                 num_classes = 3, 
                 checkpoint= 'vlm_nav/checkpoints/navigator_model.pth', 
                 device = 'cpu'):
        super(Navigator, self).__init__()
        self.device = device
        self.checkpoint = checkpoint
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16,16)
        self.fc3 = nn.Linear(16, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))      
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def save_model(self):
        torch.save(self.state_dict(), self.checkpoint)

    def load_model(self):
        self.load_state_dict(torch.load(self.checkpoint, map_location=torch.device(self.device), weights_only=True))
        
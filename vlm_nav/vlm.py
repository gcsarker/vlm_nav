from PIL import Image
import google.generativeai as genai
from google.api_core import retry
from google.generativeai.types import RequestOptions
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
key = config['Settings']['API_KEY']
genai.configure(api_key=key)

class VLM():
    def __init__(self) -> None:
        # model option
        # 1 (fastest): models/gemini-1.5-flash-latest
        # 2 (Accurate): models/gemini-1.5-pro
        self.model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        self.prompt = "Given the depthmap image from the front camera of a drone, where the darker area is closer. Which direction the drone should fly towards to avoid collision?"



    def get_vlm_feedback(self, depth):
        # Normalizing before feeding to gemini
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = depth*255
        depth = 255- depth
        depth = Image.fromarray(depth).convert('L')
        

        # gemini feedback
        feedback = self.model.generate_content(
            [self.prompt, depth],
            request_options=RequestOptions(retry=retry.Retry(initial=10, multiplier=2, maximum=60, timeout=300))
            )
        feedback = feedback.text
        feedback = feedback.split('.')[0].lower()
        direction = {
            'left': 'left' in feedback,
            'right': 'right' in feedback
            }
        
        return direction
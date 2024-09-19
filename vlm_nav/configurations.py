import configparser
config = configparser.ConfigParser()
config.read('config.ini')

key = config['Settings']['API_KEY']
api_request_delay = float(config['Settings']['API_REQUEST_DELAY'])

tau = float(config['Settings']['TAU'])
tau_d = float(config['Settings']['TAU_D'])

velocity = float(config['Settings']['VELOCITY']) #m/s
flight_height = -float(config['Settings']['FLIGHT_HEIGHT'])
yaw_angle = float(config['Settings']['YAW_ANGLE'])

flight_path_logging = True if config['Settings']['FLIGHT_LOG'] == 'True' else False
depth_estimation = True if config['Settings']['DEPTH_ESTIMATION'] == 'True' else False

map_x = float(config['Settings']['MAP_DIM_X']) - 2
map_y = float(config['Settings']['MAP_DIM_Y'])//2 - 2
forward_duration = float(config['Settings']['FORWARD_DURATION'])
open_area_forward_duration = float(config['Settings']['OPEN_SPACE_FORWARD_DURATION'])

def print_config():
    print(f"key : {key}, type = {type(key)}\n"
          f"tau : {tau}, type = {type(tau)}\n"
          f"tau_d : {tau_d}, type = {type(tau_d)}\n"
          f"velocity : {velocity}, type = {type(velocity)}\n"
          f"flight_height : {flight_height}, type = {type(flight_height)}\n"
          f"yaw rate : {yaw_angle}, type = {type(yaw_angle)}\n"
          f"flight_path_logging : {flight_path_logging}, type = {type(flight_path_logging)}\n"
          f"map dimension : ({map_x},{map_y}), type = ({type(map_x)},{type(map_y)}\n"
          f"depth_estimation : {depth_estimation}, type = {type(depth_estimation)}\n"
          f"forward_duration : {forward_duration}, type = {type(forward_duration)}\n"
          f"API call delay : {api_request_delay}, type = {type(api_request_delay)}\n"
          )

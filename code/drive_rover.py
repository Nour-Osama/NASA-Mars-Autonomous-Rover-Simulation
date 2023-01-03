# Do the necessary imports
import argparse
import shutil
import base64
from datetime import datetime
import os
import cv2
import numpy as np
import pandas as pd
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO, StringIO
import json
import pickle
import matplotlib.image as mpimg
import time

# Import functions for perception and decision making
from perception import perception_step
from decision import decision_step
from supporting_functions import update_rover, create_output_images
# Initialize socketio server and Flask application 
# (learn more at: https://python-socketio.readthedocs.io/en/latest/)
sio = socketio.Server()
app = Flask(__name__)

# Read in ground truth map and create 3-channel green version for overplotting
# NOTE: images are read in by default with the origin (0, 0) in the upper left
# and y-axis increasing downward.
ground_truth = mpimg.imread('../calibration_images/map_bw.png')
# This next line creates arrays of zeros in the red and blue channels
# and puts the map into the green channel.  This is why the underlying 
# map output looks green in the display image
ground_truth_3d = np.dstack((ground_truth*0, ground_truth*255, ground_truth*0)).astype(np.float)

# Define RoverState() class to retain rover state parameters
class RoverState():
    def __init__(self):
        self.start_time = None # To record the start time of navigation
        self.total_time = None # To record total duration of naviagation
        self.previous_time = 0 # To record time of previous image taken initially zero for first image
        self.rock_detect_time = 0 # To record time when rock is detected
        self.turning_initial_time = 0 # To record time when turning starts
        self.turning_final_time = 0 # To record time when turning finish
        self.turn_time_threshold = 5 # Threshold to indicate rover have been turning in place for a long time
        self.is_best_yaw_determined = False # Flag to indicate that best yaw is determined when turning for a long time
        self.turn_best_yaw_list = [] # list of lists with all yaw and nav_tot_ratio pairs
        self.turn_best_yaw = 0 # Best yaw angle that has teh highest nav_tot_ratio
        self.isturning = False # Flag to indicate whthear rover is turning or not
        self.img = None # Current camera image
        self.dst = 7
        self.heuristic_dst = 15
        self.scale_factor = 2
        self.scale = 0
        self.pos = None # Current position (x, y)
        self.yaw = None # Current yaw angle
        self.previous_yaw = 0 # previous yaw angle
        self.acc_yaw_rate = 0 # accumlative yaw rate that resets every fixed interval
        self.yaw_turn_thresh = 1000 # Threhold to indicate the rover is turning in a big arc
        self.yaw_sample_turn_thresh = 4000 # Threhold to indicate the rover is turning in circles in sample mode
        self.yaw_loop_thresh = 1700 # Threhold to indicate the rover is turning in circles
        self.pitch = None # Current pitch angle
        self.roll = None # Current roll angle
        self.vel = None # Current velocity
        self.steer = 0 # Current steering angle
        self.dirc = 0  # Current Rover direction that calculates steering angles
        self.dirc_num = 15 # Number of directions to consider for each image
        self.throttle = 0 # Current throttle value
        self.brake = 0 # Current brake value
        self.nav_angles = None # Angles of navigable terrain pixels
        self.nav_dists = None # Distances of navigable terrain pixels
        self.obs_angles = None # Angles of obstcale terrain pixels
        self.obs_dists = None # Distances of obstcale terrain pixels
        self.rock_angles = None # Angles of rock terrain pixels
        self.rock_dists = None # Distances of rock terrain pixels
        self.ground_truth = ground_truth_3d # Ground truth worldmap
        self.mode = 'forward' # Current mode (can be forward or stop or sample)
        self.throttle_mode = "accel" # Current throttle mode (can be accel or constant)
        self.throttle_set = 0.2 # Throttle setting when accelerating
        self.brake_set = 10 # Brake setting when braking
        # The stop_forward and go_forward fields below represent total count
        # of navigable terrain pixels.  This is a very crude form of knowing
        # when you can keep going and when you should stop.  Feel free to
        # get creative in adding new fields or modifying these!
        self.nav_tot_ratio = 0 # navigable terrain size / total terrain size
        self.forward_thresh = 0.07 # Threshold to initiate moving based on nav_tot_ratio
        self.stop_thresh = 0.05 # Threshold to initiate stopping based on nav_tot_ratio
        self.nav_obs_ratio = 0 # navigable terrain size / obstacel terrain size
        self.accel_thresh = 0.12 # Threshold to initiate accelration based on nav_obs_ratio
        self.constant_thresh = 0.07 # Threshold to initiate constant velocity based on nav_obs_ratio
        self.accel_factor = 1 # Factor to fine-tune accel value
        self.turn_obs_dist = 0 # Turning distance from obstacles to start turning away from it
        self.min_obs_dist = 0 # Minimum distance from nearest obstacle to rover
        self.min_obs_dirc = 1 # Direction of nearest obstacle to rover (+ve or -ve only)
        self.stop_dist = 0 # Rover stops when the nearest obstacle is at this distance 
        self.min_stop_dist = 1.4 # Minimum value of stop distance
        self.max_vel = 4 # Maximum velocity (meters/second)
        # Image output from perception step
        # Update this image to display your intermediate analysis steps
        # on screen in autonomous mode
        self.vision_image = np.zeros((160, 320, 3), dtype=np.float) 
        # Worldmap
        # Update this image with the positions of navigable terrain
        # obstacles and rock samples
        self.worldmap = np.zeros((200, 200, 3), dtype=np.float) 
        self.fake_worldmap = np.zeros((200, 200, 3), dtype=np.float)  # a fake world mape for processing only
        self.fake_perception_worldmap = np.zeros((200, 200, 3), dtype=np.float)  # a fake world mape for perception only
        self.samples_pos = None # To store the actual sample positions
        self.sample_yaw= 0 # Original yaw angle before entering sample mode
        self.sample_angle = 0 # last known average sample angle
        self.sample_timer = 0 # time elasped in smaple mode
        self.yaw_sample_counter = 0 # counter of turns in sample mode without moving aprox
        self.sample_cooldown  = 0 # Record time at which sample mode failed to pick up rock and wait some seconds after this time before entering smaple mode again
        self.samples_to_find = 0 # To store the initial count of samples
        self.sample_thresh = 30 # threshold to indicate a rock sample is found
        self.sample_turn_thresh  = 0 # threshold to start turning in sample mode
        self.sample_near_thresh = 100 # threshold to indicate a rock sample is very close
        self.rock_detect_initial_time = 0 # initial time when rock is detected in sample mode
        self.rock_detect_cooldown = 2.5     # time at which counter is reset in sample mode
        self.rock_detect_counter  = 0 # Counter of how mnay psuedo-consecutive rock pixels detected
        self.rock_count_limit = 20   # upper limit to how many psuedo-consecutive rock pixels before giving control back to minimum distance for movement
        self.samples_located = 0 # To store number of samples located on map
        self.samples_collected = 0 # To count the number of samples collected
        self.sample_min_dist  = 0 # Minimum distance of sample to Rover
        self.near_sample = 0 # Will be set to telemetry value data["near_sample"]
        self.picking_up = 0 # Will be set to telemetry value data["picking_up"]
        self.send_pickup = False # Set to True to trigger rock pickup
        self.debugging_path = ""  # debugging path
        self.image_df = pd.DataFrame(columns=["X_position","Y_position","yaw"],dtype=np.int8) # relative information for debugging mode

# Initialize our rover 
Rover = RoverState()
# Calculating distance from accelration = -1, initial velocity = Rover.max_vel , initial distance = 0
# the result is v(t) = 0 at t = Rover.max_vel, and d(t) =  (-t)^2/2 
# max and min distance are 3,6 respectively 
#Rover.turn_obs_dist = np.clip((Rover.max_vel * Rover.max_vel / 2),3,6)
Rover.scale = Rover.scale_factor * Rover.dst
Rover.forward_thresh *= Rover.dst/Rover.heuristic_dst
Rover.stop_thresh *= Rover.dst/Rover.heuristic_dst
Rover.accel_thresh *= Rover.dst/Rover.heuristic_dst
Rover.constant_thresh *= Rover.dst/Rover.heuristic_dst
Rover.sample_thresh *= Rover.dst/Rover.heuristic_dst 
Rover.accel_factor /= Rover.dst/Rover.heuristic_dst 
#Rover.sample_near_thresh /= Rover.dst/Rover.heuristic_dst 
print("dst:\t",Rover.dst,"\tScale:\t",Rover.scale,"\tthreshold factor:\t",Rover.dst/Rover.heuristic_dst,"\n\n\n")

# Variables to track frames per second (FPS)
# Intitialize frame counter
frame_counter = 0
# Initalize second counter
second_counter = time.time()
fps = None

# Define telemetry function for what to do with incoming data
@sio.on('telemetry')
def telemetry(sid, data):
    global frame_counter, second_counter, fps
    frame_counter+=1
    # Do a rough calculation of frames per second (FPS)
    if (time.time() - second_counter) > 1:
        fps = frame_counter
        frame_counter = 0
        second_counter = time.time()
    print("Current FPS: {}".format(fps))

    if data:
        global Rover
        Rover.debugging_path = args.image_folder 
        # Initialize / update Rover with current telemetry
        Rover, image = update_rover(Rover, data)
        if(Rover.total_time < 1):
            Rover.initial_pos = Rover.pos
        if np.isfinite(Rover.vel):
            # Execute the perception and decision steps to update the Rover's state
            Rover = perception_step(Rover)
            Rover = decision_step(Rover)
            
            # Create output images to send to server
            out_image_string1, out_image_string2 = create_output_images(Rover)

            # The action step!  Send commands to the rover!
 
            # Don't send both of these, they both trigger the simulator
            # to send back new telemetry so we must only send one
            # back in respose to the current telemetry data.

            # If in a state where want to pickup a rock send pickup command
            if Rover.send_pickup and not Rover.picking_up:
                send_pickup()
                # Reset Rover flags
                Rover.send_pickup = False
            else:
                # Send commands to the rover!
                commands = (Rover.throttle, Rover.brake, Rover.steer)
                send_control(commands, out_image_string1, out_image_string2)

        # In case of invalid telemetry, send null commands
        else:

            # Send zeros for throttle, brake and steer and empty images
            send_control((0, 0, 0), '', '')

        # If you want to save camera images from autonomous driving specify a path
        # Example: $ python drive_rover.py image_folder_path
        # Conditional to save image frame if folder was specified
        if args.image_folder != '':
            # append image info to image_df
            Rover.image_df.loc[len(Rover.image_df)] = [Rover.pos[0],Rover.pos[1],Rover.yaw]
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
           # Rover.image_df.to_csv(args.image_folder + "/image_info.csv",index=False)

    else:
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control((0, 0, 0), '', '')
    sample_data = {}
    sio.emit(
        "get_samples",
        sample_data,
        skip_sid=True)

def send_control(commands, image_string1, image_string2):
    # Define commands to be sent to the rover
    data={
        'throttle': commands[0].__str__(),
        'brake': commands[1].__str__(),
        'steering_angle': commands[2].__str__(),
        'inset_image1': image_string1,
        'inset_image2': image_string2,
        }
    # Send commands via socketIO server
    sio.emit(
        "data",
        data,
        skip_sid=True)
    eventlet.sleep(0)
# Define a function to send the "pickup" command 
def send_pickup():
    print("Picking up")
    pickup = {}
    sio.emit(
        "pickup",
        pickup,
        skip_sid=True)
    eventlet.sleep(0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()
    
    #os.system('rm -rf IMG_stream/*')
    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("Recording this run ...")
    else:
        print("NOT recording this run ...")
    
    # wrap Flask application with socketio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

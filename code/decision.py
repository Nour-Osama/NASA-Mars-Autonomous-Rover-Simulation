from perception import pix_to_world
import numpy as np
import math

def stop(Rover):
    # Set mode to "stop" and hit the brakes!
    Rover.throttle = 0
    # Set brake to stored brake value
    Rover.brake = Rover.brake_set
    Rover.steer = 0
    Rover.mode = 'stop'

# duplicate angle based on how close it is to origin (gives more weight to close angles than far ones)
def dulpicate(angles, dist):
    dist_max = dist.max()
    dist_min = dist.min()
    dup_angles = []    
    for i in range(len(dist)):
        dup = int((dist_max-dist[i])/(dist_max-dist_min) * 10)
        for j in range(dup):
            dup_angles.append(angles[i])
    return np.array(dup_angles)

# generate n (default 10) directions from an array of directions i.e angles
# by giving more weights to closer angles (angles with small distance)
def gen_dirc(dist,angles,dir_num = 10):
    dup_angles = dulpicate(angles,dist).round(2)
    dir_num = np.clip(dir_num,0,100)
    dir_list = []
    quantiles = np.linspace(0.1,0.95,dir_num)
    for quantile in quantiles:
        #TODO: could be faster by doing a single sort and then indexing based on quantile * len(dup_angles)
        dir_list.append(np.quantile(dup_angles,quantile))
    return dir_list

# calculate boundary surrounding a certain direction 
# Where the difference between the max distance between boundaries and direction is 3 meters
def calc_boundary(dirc,angles,dists,dst):
    dirc = round(dirc,2)
    distances = dists[np.where(angles==dirc)]
    distance = (distances.max()/dst) if len(distances) > 0 else 1
    shift_angle = math.atan(3/distance)
    return [dirc + shift_angle, dirc - shift_angle]

def calc_dir(dir_list,angles,dist,obs_angles,obs_dists,dst):
    crash_dir_list = []
    near_obs_dist_list = []
    dist_dir = []
    for dirc in dir_list:
        dirc_boundaries = calc_boundary(dirc,angles,dist,dst)
        near_obs_dist = obs_dists[np.where((obs_angles>=dirc_boundaries[1])&(obs_angles<=dirc_boundaries[0]))]
        near_obs_dist_list.append(near_obs_dist)
        crash_dir_list.append(len(near_obs_dist)>0)
    all_crash = True
    for crash_dir in crash_dir_list:
        all_crash = all_crash and crash_dir
    if (all_crash):
        #print("all dirc crash\n")
        for i,dirc in enumerate(dir_list):
            min_dist = near_obs_dist_list[i].min() if len(near_obs_dist_list[i]) > 0 else 0
            dist_dir.append(min_dist)  
    else:
        crash_dir_list = np.array(crash_dir_list)
        non_crash = np.where(crash_dir_list==False)
        #print(non_crash)
        dir_list = np.array(dir_list)[non_crash]
        #print("dir list",dir_list)
        for dirc in dir_list:
            max_dist_indices = np.where(angles==dirc)
         #   print(dirc,max_dist_indices)
            max_dist = dist[max_dist_indices].max() if len(dist[max_dist_indices]) > 0 else 0
            dist_dir.append(max_dist)
       # print("distance for each direction",dist_dir)
    sorted_dirc_list_idx = np.argsort(np.array(dist_dir))[::-1]
    sorted_dirc_list =  np.array(dir_list)[sorted_dirc_list_idx]
    return sorted_dirc_list

def dirc_to_angle(angles,dirc,a=-15,b=15):
    max_angles = angles.max()
    min_angles = angles.min()
    steering_angle = (b-a) * (dirc-min_angles)/(max_angles-min_angles) + a
    return steering_angle

def to_cart_coords(angles,dists):
    x = dists * np.cos(angles)
    y = dists * np.sin(angles)
    return(x, y)

def avoid_known_routes(sorted_dirc_list,angles,dists,xpos,ypos,yaw,world_size,scale,worldmap):
    #best = int(len(sorted_dirc_list)/3)
    sorted_dirc_list = sorted_dirc_list[:5] if len(sorted_dirc_list) >= 5 else sorted_dirc_list[:len(sorted_dirc_list)]
    #print(sorted_dirc_list)
    matching_percent_list= []
    nav_world_map = worldmap[:,:,2]
    for dirc in sorted_dirc_list:
        dirc = round(dirc,2)
        dirc_dists = dists[np.where(angles==dirc)]
        x,y = to_cart_coords(dirc,dirc_dists)
        x = x.round(0)
        y = y.round(0)
        x_world,y_world = pix_to_world(x, y, xpos, ypos, yaw, world_size, scale)
        matching_nav = nav_world_map[tuple([y_world,x_world])]
        matching_nav = matching_nav[matching_nav!=0]
        matching_percent = np.sum(matching_nav)/len(x) if len(x) > 0 else 500
        #print("direction percent matching",matching_percent*100)
        matching_percent_list.append(matching_percent)
    matching_percent_list = np.array(matching_percent_list)
    return sorted_dirc_list[np.argmin(matching_percent_list)] if len(matching_percent_list) > 0 else sorted_dirc_list[0]

# returns all cartisean coordinates of a direction enclosed within boundaries
def dirc_nav_cart(dirc,angles,dists,dst):
    dirc_boundaries = calc_boundary(dirc,angles,dists,dst)
    indices = np.where((angles>=dirc_boundaries[1])&(angles<=dirc_boundaries[0]))
    nav_angles = angles[indices]
    nav_dists = dists[indices]
    return to_cart_coords(nav_angles,nav_dists)

def calc_steer(Rover):
    obs_angles = Rover.obs_angles
    obs_dists = Rover.obs_dists
    angles = Rover.nav_angles
    dist = Rover.nav_dists
    dir_list = gen_dirc(dist,angles,Rover.dirc_num)
    sorted_dirc_list = calc_dir(dir_list,angles,dist,obs_angles,obs_dists,Rover.dst)
    print("Sorted Directions",sorted_dirc_list)
    # TODO: assign rover direction from the top 5 directions based on which one was least navigable before   DONE
    # pick direction that has the highest probability of being a new route
    Rover.dirc = avoid_known_routes(sorted_dirc_list,angles,dist,Rover.pos[0],Rover.pos[1],Rover.yaw,Rover.worldmap.shape[0],Rover.scale,Rover.fake_worldmap)
    # update fake world map with the selected direction navigable pixels within the boundaries 
    x_pix_boundary,y_pix_boundary = dirc_nav_cart(Rover.dirc,angles,dist,Rover.dst)
    x_nav_world, y_nav_world = pix_to_world(x_pix_boundary, y_pix_boundary,Rover.pos[0],Rover.pos[1],Rover.yaw,Rover.worldmap.shape[0],Rover.scale)
    Rover.fake_worldmap[y_nav_world, x_nav_world, 2] += 1

    min_dist = Rover.min_obs_dist
    # if rover is close to obstacle then do a sharper turn (higher absolute value of steering angle) in the same general direction
    # otherwise steering angle should be propotional to direction
    return dirc_to_angle(angles,Rover.dirc) if min_dist < Rover.turn_obs_dist else np.clip(Rover.dirc *180/np.pi, -15, 15)


def calc_throttle(Rover):
    """
     1) if navigable to obstacle ratio is above a certain threshold accelrate 
     2) if navigable to obstacle ratio is below a certain threshold constnat accel 
     3) if between two thresholds then use last known mode  (hysteresis)
     yaw factor is used to decrease accelration / increase decelration in case of a big turn
     
     accelration/deceleration value depends on:
      1) total navigable terrain ratio 
      2) a constant for fine tuning
      3) the yaw factor
    """
    # accelrate if navigable ratio is above accelration threshold 
    if Rover.nav_obs_ratio >= Rover.accel_thresh:
        Rover.throttle_mode = "accel"
    # constant velocity if navigable ratio is above decelration threshold and in constant mode
    elif  Rover.nav_obs_ratio <= Rover.constant_thresh:
        Rover.throttle_mode = "constant"
    # if rover wasn't moving add very small accelration
    if Rover.vel <=0.2: 
        Rover.throttle_mode = "accel"
        return Rover.throttle_set
    # accelrate if in accel mode 
    elif  Rover.throttle_mode == "accel":
        yaw_factor = 1 if Rover.acc_yaw_rate < Rover.yaw_turn_thresh else 0.5
        return Rover.nav_tot_ratio * Rover.accel_factor * yaw_factor
    # cosntant velocity if in constant mode
    elif Rover.throttle_mode == "constant":
        return Rover.throttle
        yaw_factor = 1 if Rover.acc_yaw_rate < Rover.yaw_turn_thresh else 1.5
        # (1- Rover.nav_tot_ratio) indicates that the ratio is based now on obstacle total ratio
        return (1-Rover.nav_tot_ratio) * Rover.accel_factor * yaw_factor  * -1 *0


def initialize_decision_vars(Rover):
    # set minimum obstacel distance to the minimum of the array if the array isn't empty
    # otherwise set it to 10
    Rover.min_obs_dist =  Rover.obs_dists.min()/Rover.dst if len(Rover.obs_dists) >0 else 10

    # set rover stop distacne to Rover.min_stop_dist + d(t) at t = Rover.vel
    # Calculating distance from accelration = -1, initial velocity = Rover.vel , initial distance = 0
    # the result is v(t) = 0 at t = Rover.vel, and d(t) =  (-t)^2/2 
    # min and max distance are Rover.min_stop_dist, 10 respectively 
    # the minimum value is the threshold to start moving again 
    Rover.stop_dist = np.clip(Rover.min_stop_dist + (Rover.vel * Rover.vel)/2,Rover.min_stop_dist,10) 
    Rover.turn_obs_dist = Rover.stop_dist * 1.3
    #sRover.stop_dist =  Rover.stop_dist * 3/2
    # round all distances and angles to 2 dp
    Rover.obs_angles = Rover.obs_angles.round(2)
    Rover.obs_dists = Rover.obs_dists.round(2)
    Rover.nav_angles = Rover.nav_angles.round(2)
    Rover.nav_dists = Rover.nav_dists.round(2)
    set_mode(Rover)

def set_mode(Rover):
    # set mode to sample if it located more rock than it collected
    # otherwise set mode to previous mode
    Rover.mode = "sample" if Rover.samples_collected < Rover.samples_located else Rover.mode
    #Rover.mode = "origin" if Rover.samples_collected >= 0 and Rover.total_time > 10 else Rover.mode
    Rover.mode = "finish" if Rover.mode == "finish" else Rover.mode
def initiate_stopping_sequence (Rover,vel,turning_cond,wait_time,mode,reverse_turn_flag):
    # if Rover velocity above certain velocity stop rover
    if Rover.vel > vel:
        stop(Rover)
    # else if velocity is below certain velocity start turning untill a certain condition is met to start moving forward again
    elif Rover.vel <= vel:
        if (turning_cond):
            Rover.throttle = 0
            # Release the brake to allow turning
            Rover.brake = 0
            Rover.turning_initial_time = Rover.turning_initial_time if Rover.isturning else Rover.total_time
            Rover.turning_final_time = Rover.total_time
            Rover.turn_best_yaw_list.append([Rover.yaw,Rover.nav_tot_ratio])
            turn(Rover)
            Rover.mode = mode
            Rover.isturning = True
        # if turning cond is false and enough waiting time have passed move forward
        elif (Rover.total_time-Rover.turning_final_time) >= wait_time :
            # Set throttle back to stored value
            Rover.throttle = calc_throttle(Rover)
            # Release the brake
            Rover.brake = 0
            # Set steering angle
            if reverse_turn_flag: 
                #update_steer(Rover)
                print("Do nothing for now")
            else : turn(Rover)
            Rover.mode = 'forward'
            Rover.isturning = False if reverse_turn_flag else Rover.isturning
# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):
    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Rover.min_obs_dist =  calc_min_obs_dist(Rover)
        # calculate minimum distance from obstacle if they are present otherwise assign a large number(10)
       # Rover.min_obs_dist =  Rover.obs_dists.min()/Rover.dst if len(Rover.obs_dists) >0 else 10
        initialize_decision_vars(Rover)
        #print("Rover Nav tot ratio, Rover nav_obs_ratio")
        #print(Rover.nav_tot_ratio,Rover.nav_obs_ratio)
        print("Mode:",Rover.mode)
        print("min dist:",  Rover.min_obs_dist)
        print("turn dist:",  Rover.turn_obs_dist)
        print("stop dist:",  Rover.stop_dist)
        #print("Rover Mode",Rover.mode)
        # Check for Rover.mode status
        if Rover.mode == "forward":   
            xo = Rover.initial_pos[0]
            yo = Rover.initial_pos[1]
            x =  Rover.pos[0]
            y =  Rover.pos[1]
            origin_dist = math.sqrt((x-xo)**2 + (y-yo)**2)
            print("\nDISTANCE FROM ORIGIN",origin_dist)
            origin_cond = Rover.samples_collected >= 4 and Rover.total_time > 1000 and origin_dist < 5
            if(origin_cond):
                stop(Rover)
                Rover.mode = "finish"
                if Rover.debugging_path != "" :Rover.image_df.to_csv(Rover.debugging_path + "/image_info.csv",index=False)
            # if rover sees enough navigable terrain and isn't turning in loops move forward and min distance from nearest obstacle above stop distance 
            #and Rover.nav_tot_ratio > Rover.stop_thresh
            elif ( Rover.nav_tot_ratio > Rover.stop_thresh and Rover.acc_yaw_rate < Rover.yaw_loop_thresh  and Rover.min_obs_dist > Rover.stop_dist ):
                update_steer(Rover)
                 #Rover.min_obs_dist > Rover.stop_dist):
                 # if rover distance fsrom nearest object is above the turn threshold accelerate
                if(Rover.min_obs_dist > Rover.turn_obs_dist or Rover.vel < 0.3):
                    accel(Rover) 
                # else if rover distance from nearest object is below turn threshold but above the stop threshold decelrate
                elif (Rover.min_obs_dist > Rover.stop_dist):
                    decel(Rover)
            # otherwise stop rover
            else:
                stop(Rover)
        # Stop mode
        elif Rover.mode == 'stop':
            print("STOP")
            vel = 0.2
            turning_cond = ( Rover.nav_tot_ratio < Rover.forward_thresh or
                 Rover.acc_yaw_rate > Rover.yaw_loop_thresh or
                 Rover.min_obs_dist < Rover.turn_obs_dist)
            wait_time = 0.3
            mode = "stop"
            reverse_turn_flag = True
            print("Turning Time",Rover.turning_final_time - Rover.turning_initial_time)
            if(Rover.turning_final_time - Rover.turning_initial_time) > 2 * Rover.turn_time_threshold:
                print("\n Move Now from STOP mode\n")
                if(not Rover.is_best_yaw_determined):
                    turn_best_yaw_list = np.array(Rover.turn_best_yaw_list)
                    Rover.turn_best_yaw = turn_best_yaw_list[turn_best_yaw_list[:,1].argmax()]
                    Rover.is_best_yaw_determined = True
                best_nav_tot_ratio = Rover.turn_best_yaw[1]
                best_yaw_cond = (Rover.yaw > Rover.turn_best_yaw[0]  + 5) or (Rover.yaw < Rover.turn_best_yaw[0] - 5 )
                nav_tot_ratio_cond = Rover.nav_tot_ratio < 0.05 * best_nav_tot_ratio
                turning_cond = best_yaw_cond or nav_tot_ratio_cond
                wait_time = 0
                reverse_turn_flag = False if nav_tot_ratio_cond else reverse_turn_flag
            elif(Rover.turning_final_time - Rover.turning_initial_time) > Rover.turn_time_threshold:
                Rover.turn_best_yaw_list.append([Rover.yaw,Rover.nav_tot_ratio])
                Rover.is_best_yaw_determined = False
            initiate_stopping_sequence (Rover,vel,turning_cond,wait_time,mode,reverse_turn_flag)
        # Sample mode
        elif Rover.mode == 'sample':
            print("Sample min dist",Rover.rock_dists.min()/Rover.dst if len(Rover.rock_dists) > 0 else 10)
            print("Rover yaw_sample_counter",Rover.yaw_sample_counter)
            if Rover.acc_yaw_rate >= (Rover.yaw_sample_turn_thresh) :
                Rover.yaw_sample_counter +=1
                Rover.acc_yaw_rate = 0
            if Rover.yaw_sample_counter == 2 or Rover.sample_timer >=60:
                Rover.yaw_sample_counter = 0
                Rover.samples_located -= 1 
                Rover.sample_cooldown = Rover.total_time
                Rover.sample_timer = 0
                #reset turning time
                Rover.turning_initial_time = Rover.total_time
            if Rover.samples_collected < Rover.samples_located:
                # assuming fps is 20 is 0.05 per frame means value is in seconds
                Rover.sample_timer += 0.05
                Rover.sample_min_dist = Rover.rock_dists.min()/Rover.dst if len(Rover.rock_dists) > 0 else 10
                vel = 0.3
                rock_cond = len(Rover.rock_dists) < Rover.sample_turn_thresh if Rover.acc_yaw_rate < Rover.yaw_sample_turn_thresh else len(Rover.rock_dists) < 1
                stop_factor = 1 - (5-Rover.sample_min_dist)/5 if Rover.sample_min_dist < 5 else 0.8
                print("stop distance after factor:",Rover.stop_dist * stop_factor)
                if not rock_cond:
                    Rover.sample_angle = np.mean(Rover.rock_angles)* 180/np.pi if len(Rover.rock_angles) > 0 else 0
                    Rover.rock_detect_initial_time = Rover.total_time if Rover.rock_detect_counter == 0 else Rover.rock_detect_initial_time
                    Rover.rock_detect_time = Rover.total_time
                    rock_detect_total_time = Rover.total_time - Rover.rock_detect_initial_time
                    print("Rock detect total time:",rock_detect_total_time)
                    Rover.rock_detect_counter =  Rover.rock_detect_counter +1 if (rock_detect_total_time < Rover.rock_detect_cooldown) else 0
                    print("Rock detect counter:",Rover.rock_detect_counter )
                many_rock_detected_cond =  Rover.rock_detect_counter > Rover.rock_count_limit 
                time_cond = (Rover.total_time - Rover.rock_detect_time) > 0.4 
                large_time_cond = (Rover.total_time - Rover.rock_detect_time) > 1
                min_dist_cond = Rover.min_obs_dist < Rover.stop_dist * stop_factor
                print("Time cond",time_cond,"Min dist cond",min_dist_cond)
                print("Sample timer",Rover.sample_timer)
                #print("length of rock.dists",len(Rover.rock_dists),"rock detect time:",Rover.rock_detect_time)
                turning_cond = (min_dist_cond or rock_cond) and (time_cond or many_rock_detected_cond)
                wait_time = 0.3
                initiate_stopping_sequence (Rover,vel,turning_cond,wait_time,"sample",True)
                if (Rover.rock_dists.min()/Rover.dst if len(Rover.rock_dists) > 0 else 10)< 1.5:
                    print("\nmove ignore risk\n")
                    Rover.throttle = 0.2 if Rover.mode != "stop" else Rover.throttle
                    Rover.steer = np.clip(Rover.sample_angle,-15,15) if Rover.mode != "stop" else Rover.steer
                # if mode is forward change steer to that of the average of the rock otherwise keep turning
                # if mode isn't forward that means that it was truning thus keeping steering angle will keep turning
                #avg_angle = np.mean(Rover.rock_angles)* 180/np.pi if len(Rover.rock_angles) > 0 else 0
                #print("Mode::::",Rover.mode)
                Rover.steer = np.clip(Rover.sample_angle,-15,15) if Rover.mode == "forward" else Rover.steer
                # make sure mode is still sample 
                Rover.mode = "sample"
                """      
                if Rover.vel > 0.3:
                    stop(Rover)
                    Rover.mode = "sample"
                else:
                    if len(Rover.rock_dists) < Rover.sample_turn_thresh:
                        Rover.brake = 0
                        turn(Rover)
                    else:
                        Rover.brake = 0
                        Rover.throttle = calc_throttle(Rover)
                        Rover.steer = np.clip(np.mean(Rover.rock_angles)* 180/np.pi,-15,15)
                """
            else:
                Rover.sample_cooldown = Rover.total_time
                Rover.sample_timer = 0
                vel = 0.2
                turning_cond = Rover.yaw > Rover.sample_yaw + 10 or Rover.yaw < Rover.sample_yaw -10
                wait_time = 0.2
                initiate_stopping_sequence (Rover,vel,turning_cond,wait_time,"sample",True)

                #print("What now ?\n")
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel <= 0.2 and not Rover.picking_up:
        Rover.send_pickup = True
    return Rover


def turn(Rover):
    # if rover has just stopped and didn't start turning
    if Rover.steer == 0:
                        # determine steering direction opposite to the direction of the closest obstacle to get away from it
        Rover.min_obs_dirc = 1 if Rover.obs_angles[np.where(Rover.obs_dists==Rover.obs_dists.min())].min() > 0 else -1              
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
    Rover.steer = -1 * Rover.min_obs_dirc * 15

def update_steer(Rover):
    # if no obstacle terrain is visible keep moving in the same direction
    if len(Rover.obs_dists) <=0:
        Rover.steer = Rover.steer
    # if no navigable terrain is visible stop 
    elif len(Rover.nav_dists) <=0:
        stop(Rover)
    # if both navigable and obstacel terrain are present calculate steer
    else: 
        Rover.steer = calc_steer(Rover)

def accel(Rover):
    if Rover.vel < Rover.max_vel:
        Rover.throttle = calc_throttle(Rover)
    else: # Else coast
        Rover.throttle = 0
def decel(Rover):
    # if rover is still moving forward decelrate
    if Rover.vel > 0:
        Rover.throttle = -1 * calc_throttle(Rover)
    else: # Else stop rover
        stop(Rover)



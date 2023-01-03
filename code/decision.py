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
def calc_dir_avoid_obs(Rover):
    dist = Rover.nav_dists
    angles = Rover.nav_angles
    obs_angles = Rover.obs_angles
    obs_dists = Rover.obs_dists
    previous_dir = Rover.dirc
    dup_angles = dulpicate(angles,dist)
    first_quartile_dir = np.quantile(dup_angles,0.35)
    mean_dir = np.mean(dup_angles)
    third_quartile_dir = np.quantile(dup_angles,0.65)
    first_quartile_dir_r = first_quartile_dir.round(2)
    mean_dir_r = mean_dir.round(2)
    third_quartile_dir_r = third_quartile_dir.round(2)
    obs_angles = obs_angles.round(2)
    obs_dists = obs_dists.round(2)
    dir_list = [first_quartile_dir_r,mean_dir_r,third_quartile_dir_r]
    crash_dir = []
    for dirc in dir_list:
        crash_dir.append(len(obs_angles[obs_angles==dirc])>0)
    # if first quartile doesn't crash but third quartile crash return first quartile
    if crash_dir[0] == False and crash_dir[2] == True : return first_quartile_dir
    # if third quartile doesn't crash but first quartile crash return third quartile
    elif crash_dir[0] == True and crash_dir[2] == False : return third_quartile_dir
    # if mean direction doesn't crash return mean direction
    elif crash_dir[1] == False : return mean_dir
    # if both first and third quartile doesn't crash but mean crashes 
    # return the one that is nearest to the previous direction 
    elif crash_dir[0] == False and crash_dir[2] == False : 
        return third_quartile_dir if (abs(first_quartile_dir - previous_dir) > abs(third_quartile_dir - previous_dir)) else first_quartile_dir
    # if all directions crash return dir with max_navigable terrain before crash
    min_dist_dir = []
    for dirc in dir_list:
        min_dist_indices = np.where(obs_angles==dirc)
        #print(max_dist_indices)
        min_dist = obs_dists[min_dist_indices].min()
        min_dist_dir.append(min_dist)
    min_dist_dir = np.array(min_dist_dir)
    best_dir_index =   np.where(min_dist_dir == min_dist_dir.max())
    return dir_list[best_dir_index[0][0]]
def calc_dir_max_navigable(Rover):
    # TODO:  replace angle max dist with len angles and try to remove the other function
    dist = Rover.nav_dists
    angles = Rover.nav_angles.round(2)
    dup_angles = dulpicate(angles,dist)
    first_quartile_dir = np.quantile(dup_angles,0.25)
    median_dir = np.quantile(dup_angles,0.5)
    third_quartile_dir = np.quantile(dup_angles,0.75)
    first_quartile_dir_r = first_quartile_dir.round(2)
    mean_dir_r = median_dir.round(2)
    third_quartile_dir_r = third_quartile_dir.round(2)
    dir_list = [first_quartile_dir_r,mean_dir_r,third_quartile_dir_r]
    max_dist_dir = []
    for dirc in dir_list:
        max_dist_indices = np.where(angles==dirc)
        #print(max_dist_indices)
        max_dist = dist[max_dist_indices].max()
        max_dist_dir.append(max_dist)
    max_dist_dir = np.array(max_dist_dir)
    best_dir_index =  np.where(max_dist_dir == max_dist_dir.max())
    return dir_list[best_dir_index[0][0]]
#def calc_dir(Rover):
def gen_dirc(dist,angles,dir_num = 10):
    dup_angles = dulpicate(angles,dist).round(2)
    dir_num = np.clip(dir_num,0,100)
    dir_list = []
    quantiles = np.linspace(0.1,0.95,dir_num)
    for quantile in quantiles:
        dir_list.append(np.quantile(dup_angles,quantile))
    return dir_list
def calc_dir(dir_list,angles,dist,obs_angles,obs_dists):
    obs_angles = obs_angles.round(2)
    obs_dists = obs_dists.round(2)
    angles = angles.round(2)
    dist = dist.round(2)
    crash_dir_list = []
    for dirc in dir_list:
        crash_dir_list.append(len(obs_angles[obs_angles==dirc])>0)
    all_crash = True
    for crash_dir in crash_dir_list:
        all_crash = all_crash and crash_dir
    dist_dir = []
    if (all_crash):
        #print("all dirc crash\n")
        for dirc in dir_list:
            dirc = dirc.round(2)
            min_dist_indices = np.where(obs_angles==dirc)
            #print(max_dist_indices)
            min_dist = obs_dists[min_dist_indices].min() if len(obs_dists[min_dist_indices]) > 0 else 0
            dist_dir.append(min_dist)
    else:
        crash_dir_list = np.array(crash_dir_list)
        non_crash = np.where(crash_dir_list==False)
        #print(non_crash)
        dir_list = np.array(dir_list)[non_crash]
        #print("dir list",dir_list)
        for dirc in dir_list:
            dirc = dirc.round(2)
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
def calc_steer(Rover):
    obs_angles = Rover.obs_angles
    obs_dists = Rover.obs_dists
    angles = Rover.nav_angles
    dist = Rover.nav_dists
    dir_list = gen_dirc(dist,angles,Rover.dirc_num)
    sorted_dirc_list = calc_dir(dir_list,angles,dist,obs_angles,obs_dists)
    print("Sorted Directions",sorted_dirc_list)
    # TODO: assign rover direction from the top 3 directions based on which one was least navigable before
    Rover.dirc = sorted_dirc_list[0]
    min_dist = Rover.min_obs_dist
    # if rover is close to obstacle then do a sharper turn (higher absolute value of steering angle) in the same general direction
    # otherwise steering angle should be propotional to direction
    return dirc_to_angle(angles,Rover.dirc) if min_dist < Rover.turn_obs_dist else np.clip(Rover.dirc *180/np.pi, -15, 15)


def calc_steer_old(Rover):
    angles = Rover.nav_angles.round(2)
    obs_dist = Rover.obs_dists
    dist = Rover.nav_dists
    # if there are navigable terrain calculate steering angle 
    # initial calculation aims to steer into driection with maximum navigable terrain 
    # regardless of obstacle position
    if len(dist) > 0:
        Rover.dirc = calc_dir_max_navigable(Rover)
        # if there are obstacle terrain and rover is below safe distance from obstacle 
        # caluclate steering angle priotrizing safety
        if len(obs_dist) > 0:
            min_dist = Rover.min_obs_dist
            #print("min dist:",min_dist)
            if min_dist < Rover.turn_obs_dist :
                Rover.dirc = calc_dir_avoid_obs(Rover)
                # clipping is used here to get as far away as possible from obstacle 
                steering_angle = np.clip(Rover.dirc *180/np.pi, -15, 15)
               # print(Rover.dirc,steering_angle)
                return steering_angle
        angles = Rover.nav_angles 
        max_angles = angles.max()
        min_angles = angles.min()
        # changing scale to a,b using Xm = (b-a) * (X- Xmin)/(Xmax-Xmin) + a
        # where a,b = -15,15
        steering_angle = 30 * (Rover.dirc-min_angles)/(max_angles-min_angles) -15
        #steering_angle = np.clip(dirc, -15, 15)
        #print(Rover.dirc,steering_angle)
        #print(steering_angle)
        return steering_angle
    # if there is no navigable terrain stop rover and return 0
    stop(Rover)
    return 0

def calc_throttle(Rover):
    """
     1) if navigable to obstacle ratio is above a certain threshold accelrate 
     2) if navigable to obstacle ratio is below a certain threshold decelrate 
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
    # decelerate if navigable ratio is above decelration threshold and in decel mode
    elif  Rover.nav_obs_ratio <= Rover.decel_thresh:
        Rover.throttle_mode = "decel"
    # if rover wasn't moving add very small accelration
    if Rover.vel <=0.2: 
        Rover.throttle_mode = "accel"
        return Rover.throttle_set
    # accelrate if in accel mode 
    elif  Rover.throttle_mode == "accel":
        yaw_factor = 1 if Rover.acc_yaw_rate < Rover.yaw_turn_thresh else 0.5
        return Rover.nav_tot_ratio * Rover.accel_factor * yaw_factor
    # decelrate if in decel mode
    elif Rover.throttle_mode == "decel":
        yaw_factor = 1 if Rover.acc_yaw_rate < Rover.yaw_turn_thresh else 1.5
        # (1- Rover.nav_tot_ratio) indicates that the ratio is based now on obstacle total ratio
        return (1-Rover.nav_tot_ratio) * Rover.accel_factor * yaw_factor  * -1

def calc_min_obs_dist(Rover):
    angles = Rover.nav_angles.round(2)
    dist = Rover.nav_dists.round(2)
    dirc= round(Rover.dirc,2)
    dst = Rover.dst
    distance_idx = dist[np.where(angles==dirc)]
    if len(distance_idx) > 0:
        distance = distance_idx.max()/dst
        shift_angle = math.atan(5/distance)
        obs_angles = Rover.obs_angles.round(2)
        obs_dists = Rover.obs_dists.round(2)
        near_obs_dist = obs_dists[np.where((obs_angles>=dirc-shift_angle)&(obs_angles<=dirc+shift_angle))]
        if (len(near_obs_dist)>0): return near_obs_dist.min()/dst
        else: return 10
    else:
        return 0
        

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
        Rover.min_obs_dist =  Rover.obs_dists.min()/Rover.dst if len(Rover.obs_dists) >0 else 10
        print("Rover Nav tot ratio, Rover nav_obs_ratio")
        print(Rover.nav_tot_ratio,Rover.nav_obs_ratio)
        print("min dist:",  Rover.min_obs_dist)
        #print("Rover Mode",Rover.mode)
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # 1) check if navigable terrain total ratio is above forward threshold
            # 2) check if Rover hasn't been turning in loops
            # 3) check if Rover isn't very close to the nearest obstacle
            #print("Rover Nav tot ratio, Rover nav_obs_ratio")
            #print(Rover.nav_tot_ratio,Rover.nav_obs_ratio)
            if ( Rover.nav_tot_ratio > Rover.stop_thresh and 
                 Rover.acc_yaw_rate < Rover.yaw_loop_thresh and
                 Rover.min_obs_dist > Rover.stop_dist):
                # if velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    Rover.throttle = calc_throttle(Rover)
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # if no obstacle terrain is visible keep moving in the same direction
                if len(Rover.obs_dists) <=0:
                    Rover.steer = Rover.steer
                # if no navigable terrain is visible stop 
                elif len(Rover.nav_dists) <=0:
                    stop(Rover)
                # if both navigable and obstacel terrain are present calculate steer
                else: 
                    Rover.steer = calc_steer(Rover) 
            # If one the three conditions is false then stop the rover
            else:
                 stop(Rover)

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if ( Rover.nav_tot_ratio < Rover.forward_thresh or
                 Rover.acc_yaw_rate > Rover.yaw_loop_thresh or
                 Rover.min_obs_dist < Rover.stop_dist):
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                else:
                    # Set throttle back to stored value
                    Rover.throttle = calc_throttle(Rover)
                    # Release the brake
                    Rover.brake = 0
                    # Set steering angle
                # if no obstacle terrain is visible keep moving in the same direction
                    if len(Rover.obs_dists) <=0:
                        Rover.steer = Rover.steer
                    # if no navigable terrain is visible stop 
                    elif len(Rover.nav_dists) <=0:
                        stop(Rover)
                    # if both navigable and obstacel terrain are present calculate steer
                    else: 
                        Rover.steer = calc_steer(Rover) 
                    Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover



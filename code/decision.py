import numpy as np


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
    previous_dir = Rover.dirc
    dup_angles = dulpicate(angles,dist)
    first_quartile_dir = np.quantile(dup_angles,0.35)
    mean_dir = np.mean(dup_angles)
    third_quartile_dir = np.quantile(dup_angles,0.65)
    first_quartile_dir_r = first_quartile_dir.round(2)
    mean_dir_r = mean_dir.round(2)
    third_quartile_dir_r = third_quartile_dir.round(2)
    obs_angles = obs_angles.round(2)
    dir_list = [first_quartile_dir_r,mean_dir_r,third_quartile_dir_r]
    crash_dir = []
    #return mean_dir
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
    # if all directions crash return dir with max_navigabel terrain
    return calc_dir_max_navigable(Rover)
def calc_dir_max_navigable(Rover):
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
    best_dir_index =   np.where(max_dist_dir == max_dist_dir.max())
    return dir_list[best_dir_index[0][0]]

def calc_steer(Rover):
    angles = Rover.nav_angles.round(2)
    obs_dist = Rover.obs_dists
    if len(obs_dist >0):
        dst = Rover.dst
        min_dist = obs_dist.min()/dst
        print("min dist:",min_dist)
        if min_dist < Rover.min_obs_dist :
            Rover.dirc = calc_dir_avoid_obs(Rover)
            # clipping is used here to get as far away as possible from obstacle 
            steering_angle = np.clip(Rover.dirc *180/np.pi, -15, 15)
            print(Rover.dirc,steering_angle)
            return steering_angle
    Rover.dirc = calc_dir_max_navigable(Rover)
    angles = Rover.nav_angles 
    max_angles = angles.max()
    min_angles = angles.min()
    # changing scale to a,b using Xm = (b-a) * (X- Xmin)/(Xmax-Xmin) + a
    # where a,b = -15,15
    steering_angle = 30 * (Rover.dirc-min_angles)/(max_angles-min_angles) -15
    #steering_angle = np.clip(dirc, -15, 15)
    print(Rover.dirc,steering_angle)
    #print(steering_angle)
    return steering_angle
    
# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:
                # if rover is turning alot (moving in circles) then stop to allow it to determine a direction
                if(Rover.acc_yaw_rate >=2500):
                    stop(Rover)
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to first quartile angle with more weight added to clsoer angles clipped to the range +/- 15
                #Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                Rover.steer = calc_steer(Rover)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
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
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steering angle
                    #Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
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


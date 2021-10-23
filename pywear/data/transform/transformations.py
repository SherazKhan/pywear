
import numpy as np
import math
import matplotlib.pyplot as plt

def calc_gravity(ax,ay,az,gravity_interval):

    ax_grav = []
    ay_grav = []
    az_grav = []

    for i in range(len(ax)):
        if(i > gravity_interval/2 and i < (len(ax) - gravity_interval/2)):
            ax_grav.append(np.mean(ax[int(i-gravity_interval/2):int(i+gravity_interval/2)]))
            ay_grav.append(np.mean(ay[int(i-gravity_interval/2):int(i+gravity_interval/2)]))
            az_grav.append(np.mean(az[int(i-gravity_interval/2):int(i+gravity_interval/2)]))
        else:
            ax_grav.append(ax[i])
            ay_grav.append(ay[i])
            az_grav.append(az[i])
    print("size of ax grav = ",len(ax_grav))
    gravity_total = []
    for i in range(len(ax_grav)):
        gravity_total.append(ax_grav[i])
    for i in range(len(ay_grav)):
        gravity_total.append(ay_grav[i])
    for i in range(len(az_grav)):
        gravity_total.append(az_grav[i])
    print("size of gravity_total = ",len(gravity_total))
    return gravity_total

def convert_to_global_frame(ax,ay,az,ax_grav,ay_grav,az_grav):
    pitch = []
    roll = []
    yaw = []
    pitch_deg = []
    roll_deg = []
    ax_global = []
    ay_global = []
    az_global = []

	#declare rotation matrix
    w, h = 3, 3;
    rotmat = [[0 for x in range(w)] for y in range(h)] 
    tmp = [0.0,0.0,0.0]

    for i in range(len(ax)):
        roll.append(math.atan2(ay_grav[i],az_grav[i]))
        pitch.append(math.atan2(-ax_grav[i],np.sqrt(ay_grav[i] * ay_grav[i] + az_grav[i] * az_grav[i])))
        yaw.append(0.0)
        phi = roll[i]
        theta = pitch[i]
        psy = yaw[i]
        pitch_deg.append(pitch[i]*(180/3.1415926))
        roll_deg.append(roll[i]*(180/3.1415926))
        rotmat[0][0] = math.cos(theta) * math.cos(psy)
        rotmat[0][1] = -1.0*math.cos(phi) * math.sin(psy) + math.sin(phi) * math.sin(theta) * math.cos(psy)
        rotmat[0][2] = math.sin(phi) * math.sin(psy) + math.cos(phi) * math.sin(theta) * math.cos(psy)
        rotmat[1][0] = math.cos(theta) * math.sin(psy)
        rotmat[1][1] = math.cos(phi) * math.cos(psy) + math.sin(phi) * math.sin(theta) * math.sin(psy)
        rotmat[1][2] = -1.0*math.sin(phi) * math.cos(psy) + math.cos(phi) * math.sin(theta) * math.sin(psy)
        rotmat[2][0] = -1.0 * math.sin(theta)
        rotmat[2][1] = math.sin(phi) * math.cos(theta)
        rotmat[2][2] = math.cos(phi) * math.cos(theta)
        tmp = np.matmul(rotmat,[ax[i],ay[i],az[i]])
        ax_global.append(tmp[0])
        ay_global.append(tmp[1])
        az_global.append(tmp[2])

    fig = plt.figure()
    plt.plot(pitch_deg,'b')
    plt.plot(roll_deg,'r')
    plt.title('blue=pitch,red=roll')
    plt.show()

    accel_global = []
    for i in range(len(ax_global)):
        accel_global.append(ax_global[i])
    for i in range(len(ay_global)):
        accel_global.append(ay_global[i])
    for i in range(len(az_global)):
        accel_global.append(az_global[i])

    return accel_global
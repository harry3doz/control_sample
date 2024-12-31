"""
steering motion model sample

reference:
https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/pure_pursuit/pure_pursuit.py

author: khiro
"""

import numpy as np
from math import cos, sin, pi, atan2, hypot
import matplotlib.pyplot as plt
from PIL import Image
import os

show_plot = True

DELTA_TIME = 0.1 # time tick [s]
SIM_TIME = 100 # simulation time [s]

Kp = 1.0  # speed proportional gain
k = 0.1  # look forward gain
LookDist = 1.5
VDesired = 10.0 

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, psi=0.0, v=0.0, psv=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.psi = psi
        self.v = v
        self.psv = psv

    def calculate_distance(self, px, py):
        dx = self.x - px
        dy = self.y - py
        return hypot(dx,dy)
class States:
    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.psi = []
        self.v = []
        self.psv = []
        self.t = []
    
    def append(self, time, state:State):
        self.t.append(time)
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.psi.append(state.psi)
        self.v.append(state.v)
        self.psv.append(state.psv)

def proportional_control(target, current):
    a = Kp * (target - current)
    return a

class RefCourse: 
    def __init__(self, cx, cy, v=0.0):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None
    def search_target_index(self, state:State):
        if self.old_nearest_point_index is None:
            dx = [state.x - icx for icx in self.cx]
            dy = [state.y - icy for icy in self.cy]
            dist = np.hypot(dx, dy)
            idx = np.argmin(dist)
            self.old_nearest_point_index = idx
        else:
            idx = self.old_nearest_point_index
            dist = state.calculate_distance(self.cx[idx], self.cy[idx])

            while True:
                dist_next = state.calculate_distance(self.cx[idx+1], self.cy[idx+1])
                if dist < dist_next:
                    break
                idx = idx + 1 if (idx+1) < len(self.cx) else idx
                dist = dist_next
            self.old_nearest_point_index = idx
        
        Lf = k * state.v + LookDist  # update 

        # search 
        while Lf > state.calculate_distance(self.cx[idx], self.cy[idx]):
            if (idx+1) >= len(self.cx):
                break
            idx +=1
        
        return idx, Lf



class SteerModel:

    def __init__(self, tr, td, wb):
        """
        Initialize two wheel model

        tr: tire radius [m]
        td: tread [m]
        wb: wheel base [m]
        """

        self.tire_radius_m = tr
        self.tread_m = td
        self.wheel_base_m = wb

    def calculate_input(self, tasr, tasl):
        """
        tasr: tire angular speed right [rad/s]
        tasl: tire angular speed left [rad/s]
        :return: speed input [m/s], yaw rate input [rad/s]
        """

        mat = np.array([[self.tire_radius_m/2, self.tire_radius_m/2],
                         [self.tire_radius_m/self.tread_m, -self.tire_radius_m/self.tread_m]])

        vec = np.array([[tasr], [tasl]])

        input_vec = np.dot(mat, vec)

        return input_vec

    def control_input(self, time, state: State, ai, alpha, di):

        v_des = ai
        psi_star = di

        #delta = self.tread_m / 2 * (v_des/self.wheel_base_m*sin(state.psi)/cos(state.psi))
        delta = self.tread_m / 2 * (v_des/self.wheel_base_m*sin(psi_star) +  alpha * cos(psi_star))
        #tasr = v_des + delta
        #tasl = v_des - delta
        tasr = v_des * abs(cos(psi_star)) + delta  # derate w.r.t. heading error
        tasl = v_des * abs(cos(psi_star)) - delta

        input_vec = self.calculate_input(tasr, tasl)
        print(input_vec, '\t', tasl, tasr)

        return input_vec

    def output_equation(self, state: State, input_vec):

        d_phi = sin(state.psi) / self.wheel_base_m

        out_mat = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0], 
                            [0, 0, 0, 1]])

        dirc_mat = np.array([[cos(state.yaw) * cos(state.psi),  0],
                             [sin(state.yaw) * cos(state.psi), 0],
                             [d_phi, 0], 
                             [-d_phi, 1]])
        
        st_mat = np.array([[state.x], [state.y], [state.yaw], [state.psi]])

        out_vec = np.dot(out_mat, st_mat) + DELTA_TIME * np.dot(dirc_mat, input_vec)
        if abs(out_vec[2,0]) > pi:
            out_vec[2,0] -=  2*pi*out_vec[2,0]/abs(out_vec[2,0])
        if abs(out_vec[3,0]) > pi/2:
            out_vec[3,0] =  pi/2*out_vec[3,0]/abs(out_vec[3,0])


        return out_vec

    def create_gif(self, im_num):
        images = []

        for num in range(im_num):
            im_name = str(num) + '.png'
            im = Image.open(im_name)
            images.append(im)
            os.remove(im_name)

        images[0].save('SteerModel.gif', save_all=True, append_images=images[1:], loop=0, duration=60)

    def pure_pursuit(self, state, trajectory: RefCourse, pidx):
        idx, Lf = trajectory.search_target_index(state)

        if pidx >= idx:
            idx = pidx
        if idx < len(trajectory.cx):
            tx = trajectory.cx[idx]
            ty = trajectory.cy[idx]
        else:  # toward goal
            tx = trajectory.cx[-1]
            ty = trajectory.cy[-1]
            idx = len(trajectory.cx) - 1
    
        alpha = atan2(ty - state.y, tx - state.x) - state.yaw
        delta = atan2(2*self.wheel_base_m*sin(alpha) / Lf, 1.0)

        return delta, alpha, idx
    
def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x,y,yaw):
            plot_arrow(ix,iy,iyaw)
    else:
        plt.arrow(x, y, length*cos(yaw), length*sin(yaw), fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def main():

    # set parameters
    tire_radius_m = 0.1
    tread_m = 0.2
    wheel_base_m = 0.6

    cx = np.arange(0, 30, 0.5)
    cy = [cos(ix / 5.0) * ix / 3.0 for ix in cx]

    # initialize
    state = State(x=0.0, y=0.0, yaw=-1.5, psi=0.0, v=0.0)
    # elapsed time
    time = 0.0
    states = States()
    states.append(time, state)
    sm = SteerModel(tire_radius_m, tread_m, wheel_base_m)
    rc = RefCourse(cx, cy)

    target_idx, _ = rc.search_target_index(state)

    while SIM_TIME >= time and target_idx < len(cx)-1:

        ai = proportional_control(VDesired, state.v)
        di, alpha, target_idx = sm.pure_pursuit(
            state, rc, target_idx
        )

        input_vec = sm.control_input(time, state, ai, alpha, di)
        st_vec = sm.output_equation(state, input_vec)

        state.x = st_vec[0,0]
        state.y = st_vec[1,0]
        state.yaw = st_vec[2,0]
        state.psv = st_vec[3,0] - state.psv
        state.psi = st_vec[3,0]
        state.v = input_vec[0,0]*cos(state.psi)

        time += DELTA_TIME
        states.append(time, state)


        print('{:.2f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(time, state.x, state.y, state.yaw, state.psi))

        if show_plot:
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event', 
                lambda event: [exit(0) if event.key == 'escape' else None])
            plot_arrow(state.x, state.y, state.yaw)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(states.x, states.y, "-b", label="trajectory")
            plt.plot(cx[target_idx], cy[target_idx], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed: " + str(state.v)[:4])
            plt.pause(0.001)

    assert target_idx <= len(cx) - 1, "Cannot goal"

    if show_plot:
        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(states.x,  states.y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(states.t, states.v, "-r")
        plt.xlabel("time[sec]")
        plt.ylabel("speed[m/sec]")
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    main()
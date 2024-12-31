"""
steering motion model sample

author: khiro
"""

import numpy as np
from math import cos, sin, pi, atan
import matplotlib.pyplot as plt
from PIL import Image
import os

show_plot = True

DELTA_TIME = 0.05 # time tick [s]
SIM_TIME = 10 # simulation time [s]

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

    def control_input(self, time, state):

        targ = 0
        if time < 5:
            targ = pi/3
        else:
            targ = -pi/3
        #targ = pi/3
        MAX_DIFF = pi/18  # 10 deg
        v_des = 1.0/self.tire_radius_m  # velocity = tire_radius * angular speed
        tasr = 0
        tasl = 0

        psi_star = targ - state[3,0]
        if abs(targ) == pi/2:
            if abs(psi_star) < 0.01:
                print('limit reached!')
                tasr = 0
                tasl = 0
            else:
                v_des = v_des * abs(atan(psi_star))
                print(state[3,0], ' --> spin turn --> ', targ)
                if psi_star >= 0:
                    tasr = v_des/10
                    tasl = -v_des/10
                else:
                    tasr = -v_des/10
                    tasl = v_des/10
        elif abs(psi_star) > MAX_DIFF:
            print(state[3,0], '-->', targ)
            if psi_star >= 0:
                tasr = v_des/5
                tasl = 0
            else:
                tasr = 0
                tasl = v_des/5
        else:
            delta = self.tread_m / 2 * (v_des/self.wheel_base_m*sin(state[3,0]) + psi_star* cos(state[3,0]))
            tasr = v_des + delta
            tasl = v_des - delta

        input_vec = self.calculate_input(tasr, tasl)
        print(input_vec, '\t', tasl, tasr)

        return input_vec

    def output_equation(self, state, input_vec):

        d_phi = sin(state[3,0]) / self.wheel_base_m

        out_mat = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0], 
                            [0, 0, 0, 1]])

        dirc_mat = np.array([[cos(state[2,0]) * cos(state[3,0]),  0],
                             [sin(state[2,0]) * cos(state[3,0]), 0],
                             [d_phi, 0], 
                             [-d_phi, 1]])

        out_vec = np.dot(out_mat, state) + DELTA_TIME * np.dot(dirc_mat, input_vec)
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

def main():
    print("Run " + __file__)

    # set parameters
    tire_radius_m = 0.1
    tread_m = 0.2
    wheel_base_m = 0.6

    # initialize
    sm = SteerModel(tire_radius_m, tread_m, wheel_base_m)

    # elapsed time
    time = 0.0

    # state vector: x, y, yaw, psi(steer angle)
    st_vec = np.zeros((4, 1))

    # figure
    ax_xy = plt.subplot(2, 1, 1)
    plt_xy, = ax_xy.plot([], [], '.', c='b', ms=10)
    ax_xy.set_xlim([-2.0, 2.0])
    ax_xy.set_ylim([-0.5, 1.0])
    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.grid(True)

    ax_tp = plt.subplot(2, 1, 2)
    plt_tp, = ax_tp.plot([], [], '.', c='b', ms=10)
    plt_ty, = ax_tp.plot([], [], '.', c='g', ms=10)
    ax_tp.set_xlim([0, SIM_TIME])
    ax_tp.set_ylim([-3.2, 3.2])
    ax_tp.set_xlabel("time")
    ax_tp.set_ylabel("b:psi[rad] / g:yaw[rad]")
    ax_tp.grid(True)

    # image number
    im_num = 0

    # simulation
    st_x = []
    st_y = []
    st_yaw = []
    st_psi = []
    #st_vec[3,0] = -pi/3
    timestamp = []

    while SIM_TIME >= time:
        time += DELTA_TIME

        input_vec = sm.control_input(time, st_vec)

        st_vec = sm.output_equation(st_vec, input_vec)

        st_x.append(st_vec[0, 0])
        st_y.append(st_vec[1, 0])
        st_yaw.append(st_vec[2, 0])
        st_psi.append(st_vec[3,0])
        timestamp.append(time)

        print('{:.2f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(time, st_vec[0,0], st_vec[1,0], st_vec[2,0], st_vec[3,0]))

        if show_plot:
            plt_xy.set_data(st_x, st_y)
            plt_tp.set_data(timestamp, st_psi)
            plt_ty.set_data(timestamp, st_yaw)
            plt.savefig(str(im_num) + '.png')
            im_num += 1
            plt.pause(0.001)

    if im_num > 0:
        sm.create_gif(im_num)

    if show_plot:
        #plt.plot(st_x, st_y, ".b")
        #plt.grid(True)
        #plt.axis("equal")
        #plt.xlabel("X [m]")
        #plt.ylabel("Y [m]")
        plt.show()

if __name__ == '__main__':
    main()
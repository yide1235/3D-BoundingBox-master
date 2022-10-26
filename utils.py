import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import glob, os

def plot_3d_box_on_3d_coor(all_pts):
    # dim = np.asarray(dim)
    # loc = np.asarray(loc)

    if not all_pts:
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for pts_origin in all_pts:
        pts_origin = np.asarray(pts_origin)

        pts = np.zeros(pts_origin.shape, dtype=pts_origin.dtype)
        pts[:, 0], pts[:, 1], pts[:, 2] = pts_origin[:, 2], pts_origin[:, 0], pts_origin[:, 1]

        # Plot figure

        ax.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2])
        ax.plot((pts[0][0], pts[1][0]), (pts[0][1], pts[1][1]), (pts[0][2], pts[1][2]), color='r')
        ax.plot((pts[0][0], pts[2][0]), (pts[0][1], pts[2][1]), (pts[0][2], pts[2][2]), color='r')
        ax.plot((pts[3][0], pts[1][0]), (pts[3][1], pts[1][1]), (pts[3][2], pts[1][2]), color='r')
        ax.plot((pts[3][0], pts[2][0]), (pts[3][1], pts[2][1]), (pts[3][2], pts[2][2]), color='r')

        ax.plot((pts[4][0], pts[5][0]), (pts[4][1], pts[5][1]), (pts[4][2], pts[5][2]), color='r')
        ax.plot((pts[4][0], pts[6][0]), (pts[4][1], pts[6][1]), (pts[4][2], pts[6][2]), color='r')
        ax.plot((pts[7][0], pts[5][0]), (pts[7][1], pts[5][1]), (pts[7][2], pts[5][2]), color='r')
        ax.plot((pts[7][0], pts[6][0]), (pts[7][1], pts[6][1]), (pts[7][2], pts[6][2]), color='r')

        ax.plot((pts[0][0], pts[4][0]), (pts[0][1], pts[4][1]), (pts[0][2], pts[4][2]), color='r')
        ax.plot((pts[1][0], pts[5][0]), (pts[1][1], pts[5][1]), (pts[1][2], pts[5][2]), color='r')
        ax.plot((pts[2][0], pts[6][0]), (pts[2][1], pts[6][1]), (pts[2][2], pts[6][2]), color='r')
        ax.plot((pts[3][0], pts[7][0]), (pts[3][1], pts[7][1]), (pts[3][2], pts[7][2]), color='r')

        # front side
        ax.plot((pts[0][0], pts[3][0]), (pts[0][1], pts[3][1]), (pts[0][2], pts[3][2]), color='b')
        ax.plot((pts[1][0], pts[2][0]), (pts[1][1], pts[2][1]), (pts[1][2], pts[2][2]), color='b')

    ax.set_xlim(40, 0)
    ax.set_ylim(-20, 20)
    ax.set_zlim(0, 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect((40,40,5))
    # ax.set_aspect(1)
    # ax.auto_scale_xyz([50,0], [-25, 25], [0, 5])
    plt.show()

def get_intrinsic_matrix_FOV_method(fov_h, fov_v, w, h):
    x = w/2
    y = h/2
    fx = x/math.tan(np.deg2rad(fov_h)/2)
    fy = y/math.tan(np.deg2rad(fov_v)/2)

    instrinsic_matrix = np.array([[fx, 0, x],
                                  [0, fy, y],
                                  [0, 0, 1]])
    return instrinsic_matrix

def get_rotation_matrix(roll, pitch, yaw):
    roll_deg, pitch_deg, yaw_deg = np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)
    roll_matrix = np.array([[1, 0, 0],
                            [0, np.cos(roll_deg), -np.sin(roll_deg)],
                            [0, np.sin(roll_deg), np.cos(roll_deg)]])
    pitch_matrix = np.array([[np.cos(pitch_deg), 0, np.sin(pitch_deg)],
                             [0, 1, 0],
                             [-np.sin(pitch_deg), 0, np.cos(pitch_deg)]])
    yaw_matrix = np.array([[np.cos(yaw_deg), -np.sin(yaw_deg), 0],
                           [np.sin(yaw_deg), np.cos(yaw_deg), 0],
                           [0, 0, 1]])
    roll_matrix[np.abs(roll_matrix) < 1e-15] = 0
    pitch_matrix[np.abs(pitch_matrix) < 1e-15] = 0
    yaw_matrix[np.abs(yaw_matrix) < 1e-15] = 0

    return np.dot(yaw_matrix, np.dot(pitch_matrix, roll_matrix))

"""
Functions to read from files
TODO: move the functions that read label from Dataset into here
"""
import numpy as np


def get_calibration_cam_to_image(cab_f):
    for line in open(cab_f):
        if 'P2:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3, 4))
            return cam_to_img

    file_not_found(cab_f)

# def get_P(cab_f):
#     for line in open(cab_f):
#         if 'P_rect_02' in line:
#             cam_P = line.strip().split(' ')
#             cam_P = np.asarray([float(cam_P) for cam_P in cam_P[1:]])
#             return_matrix = np.zeros((3,4))
#             return_matrix = cam_P.reshape((3,4))
#             return return_matrix
#
#     # try other type of file
#     return get_calibration_cam_to_image

def get_P(cab_f):
    for line in open(cab_f):
        # if 'P_rect_02' in line:
        #     cam_P = line.strip().split(' ')
        #     cam_P = np.asarray([float(cam_P) for cam_P in cam_P[1:]])
        #     return_matrix = np.zeros((3,4))
        #     return_matrix = cam_P.reshape((3,4))
        #     return return_matrix

        if 'R_02' in line:
            cam_R = line.strip().split(' ')
            cam_R = np.asarray([float(cam_R) for cam_R in cam_R[1:]])
            R = np.zeros((3, 3))
            R = cam_R.reshape((3, 3))

        if 'T_02' in line:
            cam_T = line.strip().split(' ')
            cam_T = np.asarray([float(cam_T) for cam_T in cam_T[1:]])
            T = np.zeros((3, 1))
            T = cam_T.reshape((3, 1))
            # T=T.reshape((3,1))

        if 'K_02' in line:
            cam_K = line.strip().split(' ')
            cam_K = np.asarray([float(cam_K) for cam_K in cam_K[1:]])
            K = np.zeros((3, 3))
            K = cam_K.reshape((3, 3))

    # Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    # Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    # Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0,0,1]])

    Ex = np.concatenate([R, T], axis=1)

    P = np.dot(K, Ex)

    # print(P)
    return P

    # try other type of file
    # return get_calibration_cam_to_image

def get_R0(cab_f):
    for line in open(cab_f):
        if 'R0_rect:' in line:
            R0 = line.strip().split(' ')
            R0 = np.asarray([float(number) for number in R0[1:]])
            R0 = np.reshape(R0, (3, 3))

            R0_rect = np.zeros([4,4])
            R0_rect[3,3] = 1
            R0_rect[:3,:3] = R0

            return R0_rect

def get_tr_to_velo(cab_f):
    for line in open(cab_f):
        if 'Tr_velo_to_cam:' in line:
            Tr = line.strip().split(' ')
            Tr = np.asarray([float(number) for number in Tr[1:]])
            Tr = np.reshape(Tr, (3, 4))

            Tr_to_velo = np.zeros([4,4])
            Tr_to_velo[3,3] = 1
            Tr_to_velo[:3,:4] = Tr

            return Tr_to_velo

def file_not_found(filename):
    print("\nError! Can't read calibration file, does %s exist?"%filename)
    exit()

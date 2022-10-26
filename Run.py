"""
Images must be in ./Kitti/testing/image_2/ and camera matricies in ./Kitti/testing/calib/

Uses YOLO to obtain 2D box, PyTorch to get 3D box, plots both

SPACE bar for next image, any other key to exit
"""


from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo

import os
import time

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

import argparse
import os
from utils import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument("--image-dir", default="eval/image_2/pierD/",
                    help="Relative path to the directory containing images to detect. Default \
                    is eval/image_2/")

# TODO: support multiple cal matrix input types
parser.add_argument("--cal-dir", default="eval/calib/",
                    help="Relative path to the directory containing camera calibration form KITTI. \
                    Default is camera_cal/")

parser.add_argument("--video", action="store_true",
                    help="Weather or not to advance frame-by-frame as fast as possible. \
                    By default, this will pull images from ./eval/video")

parser.add_argument("--show-yolo", action="store_true",
                    help="Show the 2D BoundingBox detecions on a separate image")

parser.add_argument("--hide-debug", action="store_true",
                    help="Supress the printing of each 3d location")


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    corners = plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes
    return location, corners

def main():
    FLAGS = parser.parse_args()

    # load torch
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        # TODO: load bins from file or something
        # model = Model.Model(features=my_vgg.features, bins=2).cuda()
        model = Model.Model(features=my_vgg.features, bins=2)
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1],map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # load yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    yolo = cv_Yolo(yolo_path)

    averages = ClassAverages.ClassAverages()

    # TODO: clean up how this is done. flag?
    angle_bins = generate_bins(2)

    image_dir = FLAGS.image_dir
    cal_dir = FLAGS.cal_dir
    if FLAGS.video:
        # if FLAGS.image_dir == "eval/image_2/" and FLAGS.cal_dir == "camera_cal/":
        #     image_dir = "eval/video/2011_09_26/image_2/"
        #     cal_dir = "eval/video/2011_09_26/"
        image_dir = "eval/video/curbside_camera/curbside_camera/"
        cal_dir = "eval/video/curbside_camera/"
    else:
        cal_dir = image_dir


    img_path = os.path.abspath(os.path.dirname(__file__)) + "/" + image_dir
    # using P_rect from global calibration file
    calib_path = os.path.abspath(os.path.dirname(__file__)) + "/" + cal_dir
    calib_file = calib_path + "calib_cam_to_cam.txt"

    # using P from each frame
    # calib_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/testing/calib/'

    try:
        ids = [x.split('.')[0] for x in sorted(os.listdir(img_path)) if x != 'calib_cam_to_cam.txt']
    except:
        print("\nError: no images in %s"%img_path)
        exit()

    for img_id in ids:

        start_time = time.time()

        img_file = img_path + img_id + ".png"

        # P for each frame
        # calib_file = calib_path + img_id + ".txt"

        truth_img = cv2.imread(img_file)
        img = np.copy(truth_img)
        yolo_img = np.copy(truth_img)

        detections = yolo.detect(yolo_img)
        all_corners = []
        for detection in detections:
            # plot_2d_box(img, detection.box_2d)
            if not averages.recognized_class(detection.detected_class):
                continue

            # this is throwing when the 2d bbox is invalid
            # TODO: better check
            try:
                detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
            except:
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = detection.box_2d
            detected_class = detection.detected_class

            # input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor = torch.zeros([1, 3, 224, 224])
            input_tensor[0,:,:,:] = input_img

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            # proj_matrix = np.hstack((np.dot(proj_matrix[:,:3], get_rotation_matrix(15,0,0)),np.zeros([3,1])))
            if FLAGS.show_yolo:
                location, corners = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
                all_corners.append(corners)
            else:
                location, corners = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)
                all_corners.append(corners)

            if not FLAGS.hide_debug:
                print('Estimated pose: %s'%location)

        if FLAGS.show_yolo:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            cv2.imwrite('predict.png', numpy_vertical)
            cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
            # plot_3d_box_on_3d_coor(all_corners)
        else:
            cv2.imshow('3D detections', img)

        if not FLAGS.hide_debug:
            print("\n")
            print('Got %s poses in %.3f seconds'%(len(detections), time.time() - start_time))
            print('-------------')

        if FLAGS.video:
            if cv2.waitKey(0) != 32: # space bar
                exit()
            # cv2.waitKey(1)
        else:
            if cv2.waitKey(0) != 32: # space bar
                exit()


if __name__ == '__main__':
    checkerboard_dir = './checkerboard/pierD/*.png'
    # mtx, dist, rvecs, tvecs = checkerboard_method(checkerboard_dir, cb_width=38)
    # mtx, dist, rvecs, tvecs = checkerboard_method(checkerboard_dir, cb_width=20)
    # R = cv2.Rodrigues(rvecs[-1])[0]
    # yawpitchrolldecomposition(R)

    main()

    # corner from lower right
    # [[-0.99747081, 0.05844907, -0.04044356],
    #  [-0.04427189, -0.95606002, -0.28980896],
    #  [-0.05560553, -0.28728547, 0.95622962]]
    # -16.722141640292172, 3.1876065363737145, -177.4586438447386

    # corner from lower right (.T)
    # [[-0.99747081, -0.04427189, -0.05560553],
    #  [0.05844907, -0.95606002, -0.28728547],
    #  [-0.04044356, -0.28980896, 0.95622962]]
    # -16.860726363213224, 2.3178773753547635, 176.64645820581376

    # Tvecs[-1]
    # [3.74127462],
    # [3.95985267],
    # [16.23030507]

    ##########################
    # corner from upper left
    # [[0.99728114, -0.05939808, -0.04361414],
    #  [0.04452076, 0.9572816, -0.28570937],
    #  [0.0587216, 0.28299083, 0.95732333]]
    # 16.468012610768298, -3.3664365410859114, 2.556109007657593

    # corner from upper left (.T)
    # [[0.99728114, 0.04452076, 0.0587216],
    #  [-0.05939808, 0.9572816, 0.28299083],
    #  [-0.04361414, -0.28570937, 0.95732333]]
    # -16.61752621968043, 2.4996991101897392, -3.4085107038172016

    # Tvecs[-1]
    # [[-3.98793383],
    # [-1.1176173 ],
    # [14.34973188]]

    # intri_m = get_intrinsic_matrix_FOV_method(117, 59, 1920, 1080)
    # print('FOV method: {}'.format(intri_m))
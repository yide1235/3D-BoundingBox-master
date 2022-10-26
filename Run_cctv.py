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
from VideoCapture import VideoCapture
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

# TODO: support multiple cal matrix input types
parser.add_argument("--cal-dir", default="eval/image_2/pierD/calib_cam_to_cam.txt",
                    help="Relative path to the directory containing camera calibration form\
                    Default is eval/image_2/pierD/calib_cam_to_cam.txt")

parser.add_argument("--rtsp", action="store_true",
                    help="whether the input data is from rtsp stream")

parser.add_argument("--video", action="store_true",
                    help="whether the input is a video")

parser.add_argument("--path", default="eval/image_2/pierD/v1.mp4",
                    help="Path to the video file or rtsp stream address. \
                    By default, this will eval/image_2/pierD/v1.mp4")

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

    calib_file = FLAGS.cal_dir

    if FLAGS.video:
        n = 0
        cap = cv2.VideoCapture(FLAGS.path)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    elif FLAGS.rtsp:
        cap = VideoCapture(FLAGS.path)
    else:
        assert False, "not video or rtsp stream, exiting"
        exit()

    vid_path, vid_writer = None, None
    while True:
        if FLAGS.video:
            n += 1
            print("Frame %i / %i" % (n, nframes))
            if n %25 != 1:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame from camera")
            break

        start_time = time.time()

        truth_img = frame
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
            numpy_vertical = cv2.resize(numpy_vertical, (1636, 1840))
            cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)

            # plot_3d_box_on_3d_coor(all_corners)

        else:
            img = cv2.resize(img, (1636, 920))
            cv2.imshow('3D detections', img)

        if FLAGS.video:
            if not vid_path:
                vid_path = 'results/video.mp4'
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()
                fourcc = 'mp4v'  # output video codec
                fps = cap.get(cv2.CAP_PROP_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*fourcc), 1, (w, h))
            vid_writer.write(cv2.resize(img, (w,h)))

        if not FLAGS.hide_debug:
            print("\n")
            print('Got %s poses in %.3f seconds'%(len(detections), time.time() - start_time))
            print('-------------')

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break


if __name__ == '__main__':
    main()
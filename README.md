# 3D Bounding Box Estimation Using Deep Learning and Geometry

## Introduction
All technical route are based on this paper. [paper](https://arxiv.org/abs/1612.00496).

## Requirements
- Python 3.9
- Torch 1.11.0
- Torchvision 0.12.0
- OpenCV 4.6.0

## Usage
In order to download the weights:
```
cd weights/
./get_weights.sh
```
This will download pre-trained weights for the 3D BoundingBox net and also YOLOv3 weights from the
official yolo [source](https://pjreddie.com/darknet/yolo/).

>If script is not working: [pre trained weights](https://drive.google.com/open?id=1yEiquJg9inIFgR3F-N5Z3DbFnXJ0aXmA) and 
[YOLO weights](https://pjreddie.com/media/files/yolov3.weights)

# How to run our model:
Important! Before running, make sure you have a camera calibration file for the camera you want to test, we have provided the default calibration file for the camera in Pier D gate D66. You can change the file by adding the argument ‘‘--cal-dir {path to camera calibration file}’ at the end of command ‘python Run_cctv.py’. To learn how to get the camera calibration, jump to section Technical details: camera calibration section.

## Test on sample videos from Pier D camera
```
python Run_cctv.py --video --path eval/image_2/pierD/v1.mp4
```
## Test on rtsp live stream of Pier D camera
```
python Run_cctv.py --rtsp --path rtsp://{account}:{password}@{ip}:{port}
```
Please acquire the rtsp info from Pawel

## Argument explanation:
```
--cal-dir {path}, change path of calibration file
--rtsp, whether the input is rtsp stream
--video, whether the input is a video
--path {path}, path for the video or rtsp address
--hide-debug, not printing location for each object 
```
Get more detail from Run_cctv.py



## Training
First, the data must be downloaded from [Kitti](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
Download the left color images, the training labels, and the camera calibration matrices. Total is ~13GB.
Unzip the downloads into the Kitti/ directory.

```
python Train.py
```
By default, the model is saved every 10 epochs in weights/.
The loss is printed every 10 batches. The loss should not converge to 0! The loss function for
the orientation is driven to -1, so a negative loss is expected. The hyper-parameters to tune
are alpha and w (see paper). I obtained good results after just 10 epochs, but the training
script will run until 100.

## How it works
The PyTorch neural net takes in images of size 224x224 and predicts the orientation and
relative dimension of that object to the class average. Thus, another neural net must give
the 2D bounding box and object class. I chose to use YOLOv3 through OpenCV.
Using the orientation, dimension, and 2D bounding box, the 3D location is calculated, and then
back projected onto the image.

There are 2 key assumptions made:
1. The 2D bounding box fits very tightly around the object
2. The object has ~0 pitch and ~0 roll (valid for cars on the road)


# Face Mask Detection with YOLO

<a href="https://colab.research.google.com/github/MINED30/Face_Mask_Detection_YOLO/blob/main/yololomask.ipynb" target="_parent\"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- Easily try YOLO v3 (v4 as well) 
- Powered by google colab
- High accuracy (mAP@0.50 = 94.04 %)
- Detecting in 3 classes

You only look once, or YOLO, is one of the faster object detection algorithms out there. Though it is no longer the most accurate object detection algorithm, it is a very good choice when you need real-time detection, without loss of too much accuracy. 

A few weeks back, the third version of YOLO came out, and this post aims at explaining the changes introduced in YOLO v3. This is not going to be a post explaining what YOLO is from the ground up. I assume you know how YOLO v2 works. If that is not the case, I recommend you to check out the following papers by Joseph Redmon et all, to get a hang of how YOLO works.

## Dataset
The dataset for this pre-trained network is provided by [VictorLin000](https://github.com/VictorLin000/YOLOv3_mask_detect) and contains 678 images of people with and without masks. In total there are 3 different classes annotated:

* `no mask` - No mask at all.
* `improperly` - Partial covered face.
* `mask` - Mask coveres the essential parts.

You can download the dataset directly from [google drive](https://drive.google.com/drive/folders/1aAXDTl5kMPKAHE08WKGP2PifIdc21-ZG).

## Model structure
<img src="https://github.com/MINED30/Face_Mask_Detection_YOLO/blob/main/demo/model.png?raw=true">

We use a new network for performing feature extraction.
Our new network is a hybrid approach between the network
used in YOLOv2, Darknet-19, and that newfangled residual
network stuff. Our network uses successive 3 × 3 and 1 × 1
convolutional layers but now has some shortcut connections
as well and is significantly larger. It has 53 convolutional
layers so we call it Darknet-53!

Darknet-53
YOLO v2 used a custom deep architecture darknet-19, an originally 19-layer network supplemented with 11 more layers for object detection. With a 30-layer architecture, YOLO v2 often struggled with small object detections. This was attributed to loss of fine-grained features as the layers downsampled the input. To remedy this, YOLO v2 used an identity mapping, concatenating feature maps from from a previous layer to capture low level features.
However, YOLO v2’s architecture was still lacking some of the most important elements that are now staple in most of state-of-the art algorithms. No residual blocks, no skip connections and no upsampling. YOLO v3 incorporates all of these.
First, YOLO v3 uses a variant of Darknet, which originally has 53 layer network trained on Imagenet. For the task of detection, 53 more layers are stacked onto it, giving us a 106 layer fully convolutional underlying architecture for YOLO v3. This is the reason behind the slowness of YOLO v3 compared to YOLO v2. Here is how the architecture of YOLO now looks like.

## Demo
<img src="https://github.com/MINED30/Face_Mask_Detection_YOLO/blob/main/demo/ezgif-2-6ce114bf9fed.gif?raw=true" width="60%">
<img src="https://github.com/MINED30/Face_Mask_Detection_YOLO/blob/main/demo/predictions%20(2).jpg?raw=true" width="60%">
<img src="https://github.com/MINED30/Face_Mask_Detection_YOLO/blob/main/demo/ezgif-3-6e175c3b97a8.gif?raw=true" width="60%">
<img src="https://raw.githubusercontent.com/MINED30/Face_Mask_Detection_YOLO/main/demo/predictions%20(1).jpg" width="60%">

## Evaluation
### Average Precision
| class_id | name       | TP  | FP | ap     |
|----------|------------|-----|----|--------|
| 0        | mask       | 333 | 34 | 96.96% |
| 1        | improperly | 12  | 6  | 92.80% |
| 2        | no mask    | 62  | 13 | 92.37% |

### F1-score & Average IoU
| conf_thresh | precision | recall | F1-score | TP  | FP | FN | average IoU |
|-------------|-----------|--------|----------|-----|----|----|-------------|
| **0.25**        |    0.88   |  0.95  |   0.92   | 407 | 53 | 21 |   69.55 %   |

### Mean Average Precision
mean average precision (mAP@0.50) = 94.04 % 

## Usage
### Load weight
```
!wget https://pjreddie.com/media/files/darknet53.conv.74
```

or you can get pretrained [weights](https://drive.google.com/drive/folders/16MsdDvPuF6CxFd0vW2VYya6e5cqZJjZI?usp=sharing)
for this data
### Train
```
!./darknet detector train /content/YOLOv3_mask_detect/Mask/object.data\
                          /content/YOLOv3_mask_detect/Mask/detect_mask.cfg\
                          darknet53.conv.74\
                          -dont_show -map 
```
### Detect
```
!./darknet detector test /content/YOLOv3_mask_detect/Mask/object.data\
                         /content/YOLOv3_mask_detect/Mask/detect_mask.cfg\
                         /content/backup/detect_mask_last.weights\
                         /content/gdrive/MyDrive/mask_demo/man_0_1.png
```

## Reference
- https://arxiv.org/abs/1804.02767
- https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b
- https://github.com/AlexeyAB/darknet
- https://github.com/VictorLin000/YOLOv3_mask_detect
- https://pjreddie.com/darknet/yolo/
- https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg
- https://medium.com/@artinte7
- https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
- https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/

## Source from
- https://pixabay.com/
- https://www.miricanvas.com/
- https://www.videvo.net/

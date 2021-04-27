<a href="https://github.com/MINED30/Face_Mask_Detection_YOLO/blob/main/ReadMeKR.md" target="_parent\">[한글 설명]</a>
# Face Mask Detection with YOLO :mask:<br><a href="https://colab.research.google.com/github/MINED30/Face_Mask_Detection_YOLO/blob/main/yololomask.ipynb" target="_parent\"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  

### *Feature*         

:rocket: Easily try YOLO v3 (v4 as well) 

:zap: Powered by google colab 

:thumbsup: High accuracy (mAP@0.50 = 94.04 %)

:three: Detecting in 3 classes

YOLO(You only look once) is one of the fastest object detection algorithms. Although it is no longer the most accurate object detection algorithm, it is recommended that you choose when real-time detection is needed without losing too much accuracy. In this repository, we will try YOLO v3 & v4 and detect facial mask.

## Dataset
The dataset for this pretrained network is provided by [VictorLin000](https://github.com/VictorLin000/YOLOv3_mask_detect) and contains 678 images of people with and without masks. There are 3 different classes annotated:

* `no mask` - No mask at all.
* `improperly` - Partially covered face.
* `mask` - Mask covers the essential parts.

You can download the dataset directly from [google drive](https://drive.google.com/drive/folders/1aAXDTl5kMPKAHE08WKGP2PifIdc21-ZG).

## Model structure
<img src="https://github.com/MINED30/Face_Mask_Detection_YOLO/blob/main/demo/model.png?raw=true">

Darknet-53 network uses successive 3 × 3 and 1 × 1 convolutional layers but now has some shortcut connections as well and is significantly larger. It has 53 convolutional layers.
YOLO v2 originally used a custom deep architecture darknet-19 which has 19 layers with 11 additional layers for object detection. YOLO v2 often suffers from small object detection. This is because the layers downsampled the input. To solve this problem, YOLO v2 used an identity mapping to link the feature maps of the previous layer to capture lower-level features (which is able to reflect small object feature).
However, the architecture of YOLO v2 still lacked the most important elements required by detecting small object. YOLO v3 puts it all together.
First, YOLO v3 uses a variant of Darknet with 53 layer networks trained on "Imagnet". For detection tasks, 53 more layers are stacked, so YOLO v3 has a 106 layer fully convolution. That's why YOLO v3 is slower than YOLO v2. 


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
!./darknet detector train /content/Face_Mask_Detection_YOLO/Mask/object.data\
                          /content/Face_Mask_Detection_YOLO/Mask/detect_mask.cfg\
                          darknet53.conv.74\
                          -dont_show -map 
```
### Detect
```
!./darknet detector test /content/Face_Mask_Detection_YOLO/Mask/object.data\
                         /content/Face_Mask_Detection_YOLO/Mask/detect_mask.cfg\
                         /content/backup/detect_mask_last.weights\
                         /content/gdrive/MyDrive/mask_demo/man_0_1.png
```

## Reference
- https://arxiv.org/abs/1804.02767
- https://arxiv.org/abs/1506.02640 
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

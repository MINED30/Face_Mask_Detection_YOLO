# Face Mask Detection with YOLO

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
- Easily use YOLO v3 with google colab
- Hello
- World✨

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

## Demo
<img src="https://github.com/MINED30/Face_Mask_Detection_YOLO/blob/main/demo/predictions%20(2).jpg?raw=true" width="60%">
<img src="https://raw.githubusercontent.com/MINED30/Face_Mask_Detection_YOLO/main/demo/predictions%20(1).jpg" width="60%">



## Performance
### one
| class_id | name       | TP  | FP | ap     |
|----------|------------|-----|----|--------|
| 0        | mask       | 333 | 34 | 96.96% |
| 1        | improperly | 12  | 6  | 92.80% |
| 2        | no mask    | 62  | 13 | 92.37% |

### two
| conf_thresh | precision | recall | F1-score | TP  | FP | FN | average IoU |
|-------------|-----------|--------|----------|-----|----|----|-------------|
| **0.25**        |    0.88   |  0.95  |   0.92   | 407 | 53 | 21 |   69.55 %   |

### Mean Average Precision
mean average precision (mAP@0.50) = 94.04 % 

## Reference
https://arxiv.org/abs/1804.02767

https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b

## Source from
https://pixabay.com/

https://www.miricanvas.com/

https://www.videvo.net/

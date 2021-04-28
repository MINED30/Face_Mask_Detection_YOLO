# Face Mask Detection with YOLO :mask:<br><a href="https://colab.research.google.com/github/MINED30/Face_Mask_Detection_YOLO/blob/main/yololomask.ipynb" target="_parent\"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### *Feature*         

:rocket: 쉽게 YOLO v3를 사용해볼 수 있습니다 (v4도 가능합니다.) 

:zap: 구글 코랩을 사용하여 언제 어디서나 사용할 수 잇습니다.

:thumbsup: 정확도가 높습니다! (mAP@0.50 = 94.04 %)

:three: 3개로 분류합니다.

YOLO(You only look once)는 객체 감지 알고리즘 중 매우 빠른 알고리즘으로 알려져있습니다. 가장 정확한 개체 탐지 알고리즘은 아니지만, 정확도가 크게 떨어지지 않으면서 실시간 탐지가 필요하다면 욜로가 좋은 선택이 될 수 있습니다. 여기서는 YOLO v3와 v4를 사용해보고 안면 마스크 착용여부를 식별해보겠습니다. YOLO가 익숙하지 않으시거나 처음이신 분들께 도움이 되었으면합니다.

## Dataset
사전훈련된 모델은 [VictorLin000](https://github.com/VictorLin000/YOLOv3_mask_detect)님으로부터 만들어졌으며,추가적인 훈련을통해 성능을 향상시켰습니다. 이 데이터셋에서는 세 가지로 분류합니다.

* `no mask` - 마스크를 쓰지 않은 사람
* `improperly` - 일부만 착용한 사람
* `mask` - 마스크를 잘 착용한 사람

데이터셋은 구글드라이브에서 직접 받을 수 있습니다. [google drive](https://drive.google.com/drive/folders/1aAXDTl5kMPKAHE08WKGP2PifIdc21-ZG).

## Model structure
<img src="https://github.com/MINED30/Face_Mask_Detection_YOLO/blob/main/demo/model.png?raw=true">


Darknet-53은 연속적인 3 × 3 및 1 × 1인 convolution layer를 사용하며, shorcut connection을 갖고있습니다. 기존보다 훨씬 더 크며, 53개의 convolution layer를 가지고 있습니다.

기존 YOLO v2는 19개의 레이어와 11개의 추가 레이어를 사용하는 darknet-19를 사용했습니다. YOLO v2는 작은 개체를 잘 감지하지 못하는 문제가 있었습니다. 다운샘플링을 했기 때문입니다. 이 문제를 해결하기 위해 YOLO v2는 identity mapping을 사용하여 이전 계층의 feature map과 연결하였고, lower-level features(작은 개체의 특징이 잘 드러남)를 반영하였습니다. 하지만 YOLO v2는 작은 개체를 감지하는데에 있어 구조상으로 문제가 있었습니다. YOLO v3는 이 문제를 어느정도 보완합니다. v2와 비교하면 v3가 컨볼루션이 더 많아졌기 때문에, 속도는 더 느립니다.

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

이 데이터셋으로 훈련된 [weights](https://drive.google.com/drive/folders/16MsdDvPuF6CxFd0vW2VYya6e5cqZJjZI?usp=sharing)를
다운로드해서 사용할 수 있습니다.

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
                         /content/Face_Mask_Detection_YOLO/demo/man_0_1.png
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

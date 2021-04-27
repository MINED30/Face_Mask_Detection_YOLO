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
darknet-53

## Demo
![image](https://user-images.githubusercontent.com/73981982/116174389-478ff500-a749-11eb-8315-7b40014fd230.png)



## Performance
class_id = 0, name = mask, ap = 96.96%     (TP = 333, FP = 34) 
class_id = 1, name = improperly, ap = 92.80%     (TP = 12, FP = 6) 
class_id = 2, name = no mask, ap = 92.37%      (TP = 62, FP = 13) 
for conf_thresh = 0.25, precision = 0.88, recall = 0.95, F1-score = 0.92 
for conf_thresh = 0.25, TP = 407, FP = 53, FN = 21, average IoU = 69.55 % 
mean average precision (mAP@0.50) = 0.940448, or 94.04 % 
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

# GuyFoxMask-Glasses Detection
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1stJa6AyzndZ2fsadl1cg_sU5tyGXcVTp?usp=sharing)
## Structure

- Dataset collection
- Annotation
- Training Mask R-CNN
- Training YOLOv8-nano
- Models evaluation

## Dataset collection

Dataset 'GuyFoxMask-Glasses' was collected in Innopolis University using smartphone realme 6 pro. Dataset consists of 2 object instances 'GuyFoxMask' and 'Glasses'. Overall number of images is 92.
![alt text](https://github.com/Extrabution/GuyFoxMask-Glasses-Detection/blob/main/images/RoboflowDataset.png?raw=true)

## Annotation

To annote 92 images I used robflow. By the end of annotation process I got 107 instancess of 2 objects. 50 for GuyFoxMask and 57 for Glasses
I've split the data set to train|val|test in folowwing proportion 70%|18%|12% or 64|17|11 images.
![alt text](https://github.com/Extrabution/GuyFoxMask-Glasses-Detection/blob/main/images/RoboflowAnnotate.png?raw=true)

## Training Mask-R-CNN

For training Mask-R-CNN I used detectron2 pre-trained model (mask_rcnn_R_50_FPN_3x). I hyper tuned batch size(set to 6) and set number of iteraions to 600. This gave me the following results:

![alt text](https://github.com/Extrabution/GuyFoxMask-Glasses-Detection/blob/main/images/FasterRCNN.png?raw=true)

## Training YOLOv8-nano-seg
For training YOLOv8-nano-seg I used ultralytics pre-trained model. I hyper tuned batch size(set to 32) and set number of epochs to 256. This gave me the following results:

![alt text](https://github.com/Extrabution/GuyFoxMask-Glasses-Detection/blob/main/images/Yolov8NanoResults.png?raw=true)


## Models evaluation

#### yolo-v8-nano:
  - train time 15:43 for 185 epochs
  - inference ~15ms
  - size ~6mb
  - map ~ 0.58

#### Mask R-CNN:
 - train time 11:43 on Tesla T4 GPU
 - inference ~ 93ms on Tesla T4 GPU
 - size ~ 314mb
 - map ~ 0.62 

 We can see that YOLO is way faster in train time and inference. Size of the weights of the model is smaller too (simply because we used nano version of the model) Map results is quite interesting as for yolo map is ~0.9 and for faster rcnn is 0.6, even though faster rcnn showed better results on the test dataset(recognized objects more precisely).

 In colnlusion, Yolo is very light detection model, that shows great results on small datasets, while faster rcnn is a heavy model and shows good results too.
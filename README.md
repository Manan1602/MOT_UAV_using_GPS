# MOT using GPS (Yolov7 + StrongSORT with OSNet + GPS Estimation)


## Introduction
This is a method to track objects through UAVs. It uses [yolov7](https://github.com/WongKinYiu/yolov7) as the object detector and uses the methodoly as proposed in [Memory Maps for Video Object Detection and Tracking on UAVs](https://arxiv.org/pdf/2303.03508v1.pdf) to estimate the GPS coordinates using the metadata collected through the drones. Finally it uses [StrongSORT](https://github.com/dyhBUPT/StrongSORT) [(paper)](https://arxiv.org/pdf/2202.13514.pdf) in order to perform object reidentification and tracking. The code is augmented to use distance between the GPS coordinates obtained (in addition to appearance features) instead of on frame distances for reidentification of objects.
## Before you run the tracker

1. Clone the repository recursively:

`git clone --recurse-submodules https://github.com/Manan1602/MOT_UAV_using_GPS.git`

or

`git clone https://github.com/Manan1602/MOT_UAV_using_GPS.git`
`git submodule update --init --recursive`

<!-- `pip install -r requirements.txt` -->


## Tracking sources

Tracking can be run on only one format for now

```bash
$ python track.py --source path/  # directory
```

### Images Format
Note that the directory must have frames in one of the following formats:
>- bmp
>- jpg
>- jpeg
>- png
>- tif
>- tiff
>- dng
>- webp
>- mpo
>
 The frames **must** also be named as frame_no.{jpg} and must also be in order for the code to work properly.
### GPS data format
GPS data for each frame must be stored in a .txt file in the same directory named the same as the frame. I.e. for frame 0.jpg there must be a file 0.txt containing the GPS data

The txt file should be comma seperated and have values in the following format:
>gps_latitude,gps_latitude_ref,gps_longitude,gps_longitude_ref,altitude,gimbal_pitch,camera_heading,gimbal_heading,focal_length
>
>47.671755,N,9.269907,E,11.299448948491314,45.4,319.3,322.4,15

Note that even if your data has some additional headings, it's fine to have them too as long as above mentioned headings are present.

Gps info of each frame should only have 2 lines, 1 for headings and 1 for values.

This code of created with consideration of a gimball camera and hence doesn't need code for camera pitch correction with respect to the UAV.

**If your data is in the same format as [SeaDronesSee MOT](https://seadronessee.cs.uni-tuebingen.de/dataset) then you can directly run `python extract_gps.py` to extract gps information and store it in the desired format.**
## Select object detection and ReID model

### Yolov7

There is a clear trade-off between model inference speed and accuracy. In order to make it possible to fulfill your inference speed/accuracy needs
you can select a Yolov7 family model for automatic download

```bash


$ python track.py --source path/ --yolo-weights yolov7.pt --img 640
                                            yolov7x.pt --img 640
                                            yolov7-e6e.pt --img 1280
                                            ...
```

Or for custom dataset, just place your pretrained yolo model in the root directory and simply run:

```bash
$ python track.py --source path/ --yolo-weights weights.pt --img img_size 
```

### StrongSORT

The above applies to StrongSORT models as well. Choose a ReID model based on your needs from this ReID [model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)

```bash


$ python track.py --source 0 --strong-sort-weights osnet_x0_25_market1501.pt
                                                   osnet_x0_5_market1501.pt
                                                   osnet_x0_75_msmt17.pt
                                                   osnet_x1_0_msmt17.pt
                                                   ...
``` 
#### Strong Sort Parameters
The parameters for StrongSort are stored in [stong_sort/configs/strong_sort.yaml](https://github.com/Manan1602/MOT_UAV_using_GPS/blob/main/strong_sort/configs/strong_sort.yaml) and can be changed as per requirements.
The parameters are:
> ECC : [ bool ] 
>> For camera motion compensation
>
> MC_LAMBDA : [ float (0-1) ]
>> It is the contribution of distance in the cost matrix
>>
>> $cost\_ matrix = \lambda * dist\_ cost\_ matrix  + (1-\lambda)*appearance\_ cost\_ matrix$
>
> EMA_ALPHA: [ float (0-1) ] updates appearance state in an exponential manner
>> $e_i^t = \alpha e_i^{t-1} + (1-\alpha)f_i^t$
>
> MAX_DIST: [ float ] 
>> The matching threshold for appearance. Samples with larger distance are considered an invalid match
>
> MAX_DIST_POS: [ float ] 
>> Matching threshold for position.
>
> MAX_IOU_DISTANCE [ float ]
>> Gating threshold. Associations with cost larger than this value are disregarded.
>
> MAX_AGE: [ int ]
>> Max frames of absence before a track is deleted.
>
> N_INIT: [ int ]
>> Number of frames that a track remains in initialization phase
>
>NN_BUDGET: [ int ]
>> Maximum size of the appearance descriptors gallery

## Filter tracked classes

By default the tracker tracks all MS COCO classes.

If you want to track a subset of the MS COCO classes, add their corresponding index after the classes flag

```bash
python track.py --source 0 --yolo-weights yolov7.pt --classes 16 17  # tracks cats and dogs, only
```

[Here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) is a list of all the possible objects that a Yolov7 model trained on MS COCO can detect. Notice that the indexing for the classes in this repo starts at zero.


## MOT compliant results

Can be saved to your experiment folder `runs/track/<yolo_model>_<deep_sort_model>/` by 

```bash
python track.py --source ... --save-txt
```


## Attribution

A big part of this code can be attributed to https://github.com/mikel-brostrom/Yolov7_StrongSORT_OSNet.

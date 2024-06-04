1. First move into the yolov7 directory
```bash
$ cd yolov7
```
2. Now edit the file at [data/custom.yaml](data/custom.yaml) and type in the location of your data in accordance to yolo format.
3. Our custom architecture combining yolov7 and swin transformers is present at [cfg/training/yolo7_custom.yaml](cfg/training/yolo7_custom.yaml).
4. For changing the hyperparameters make changes in [data/hyp.scratch.custom.yaml](data/hyp.scratch.custom.yaml)
5. Make sure your images are named by their frame_number.{jpg} without any other characters to enable the dataloader to load them in order
    This is needed to preserve temporal information.
6. Simply run the training loop as:
``` bash
$ python train.py --cfg cfg/training/yolo7_custom.yaml --data data/custom.yaml --hyp data/hyp.scratch.custom.yaml --epochs [num_epochs] --batch-size [batch_size] --img-size img_size
```
This also supports wandb logging
simply add the information in with the --project, --entity and --name tags

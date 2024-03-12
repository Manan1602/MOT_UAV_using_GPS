import argparse
import cv2
from pathlib import Path
import torch
import os
import torchvision.transforms.functional as TF
import sys
import random
FILE = Path(__file__).resolve()
print('file:')
print(FILE)
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size,scale_coords, non_max_suppression
from yolov7.utils.datasets import LoadImages
from yolov7.utils.plots import plot_one_box

def run(yolo_weights,img_dir,img_size,show_vid=False,save_vid = False,conf_thres=0.25,iou_thres=0.45,classes=None,agnostic_nms=False):
    device = torch.device('0') if torch.cuda.is_available() else torch.device('cpu')
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    model = attempt_load(yolo_weights, map_location=device)
    names, = model.names,
    stride = model.stride.max().cpu().numpy()  # model stride
    imgsz = check_img_size(img_size[0], s=stride)
    dataset = LoadImages(img_dir,imgsz,stride)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    nr_sources = 1
    for (path, im, im0s, vid_cap) in dataset:
        im1 = im0s.copy()
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]
        print('-------')
        pred = model(im)
        pred = non_max_suppression(pred[0], conf_thres, iou_thres, classes, agnostic_nms)
        print('-------')
        print(pred)
        # im1 = cv2.resize(im1,(640,640))
        pred[0][:, :4] = scale_coords(im.shape[2:], pred[0][:, :4], im1.shape).round()
        for det in pred[0]:
            
            bboxes = det[:4]
            conf = det[4]
            cls = det[5]
            print(cls)
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(bboxes,im1,label = label, color = colors[int(cls)], line_thickness = 2)
            
        
        cv2.imshow('img',cv2.resize(im1,(1200,800)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        break


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str)
    parser.add_argument('--img-dir', type=str, default = '0')
    parser.add_argument('--img-size', nargs = '+', type = int, default = [640])
    parser.add_argument('--show-vid', action = 'store_true')
    parser.add_argument('--save-vid', action = 'store_true')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
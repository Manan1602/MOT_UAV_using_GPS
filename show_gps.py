import argparse
import cv2
from tqdm import tqdm
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
from custom_dataloader import LoadImages
from yolov7.utils.plots import plot_one_box
from gps_estimation import absolute_coordinates

def run(yolo_weights,img_dir,img_size,save_dir,show_vid=False,save_vid = False,conf_thres=0.25,iou_thres=0.45,classes=None,agnostic_nms=False ):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device('0') if torch.cuda.is_available() else torch.device('cpu')
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    model = attempt_load(yolo_weights, map_location=device)
    names, = model.names,
    stride = model.stride.max().cpu().numpy()  # model stride
    imgsz = check_img_size(img_size[0], s=stride)
    dataset = LoadImages(img_dir,imgsz,stride)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for path, im, im0s, gps_info, vid_cap in tqdm(dataset):
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
        for i,det in enumerate(pred):
            p, im1, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)
            txt_file_name = p.parent.name  # get folder name containing current img
            save_path = str(Path(save_dir) / p.parent.name)
            if det is not None and len(det):
                det[:,:4] = scale_coords(im.shape[2:], det[:,:4], im1.shape).round()
                x_center = (det[:,0] + det[:,2])/2
                x_center = x_center.detach().cpu().numpy()
                y_center = (det[:,1]+det[:,3])/2
                y_center = y_center.detach().cpu().numpy()
                gps_coords = absolute_coordinates(im1.shape,gps_info['gimbal_heading'].to_numpy(),gps_info['altitude'].to_numpy(),15,gps_info['gimbal_pitch'].to_numpy(), x_center, y_center, gps_info['gps_latitude'].to_numpy(), gps_info['gps_longitude'].to_numpy())
                print(gps_info['gimbal_heading'].to_numpy())
                print(len(gps_coords))
                print(gps_coords)
                for d,gps in zip(det,gps_coords):
                    bbox = d[:4]
                    conf = d[4]
                    cls = d[5]
                    label = f"{names[int(cls)]} {conf:.2f} {gps[0]} {gps_info['gps_latitude_ref'][0]} {gps[1]} {gps_info['gps_longitude_ref'][0]}"
                    plot_one_box(bbox,im1,label = label, color = colors[int(cls)], line_thickness = 2)
        # im1 = cv2.resize(im1,(640,640))
        if show_vid:
            cv2.imshow('img',cv2.resize(im1,(1200,800)))
            cv2.waitKey(1)
        if save_vid:
            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im1.shape[1], im1.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[i].write(im1)
        prev_frames[i] = curr_frames[i]
        if save_vid:
            print(f"Results saved to { save_dir}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str)
    parser.add_argument('--img-dir', type=str, default = '0')
    parser.add_argument('--img-size', nargs = '+', type = int, default = [640])
    parser.add_argument('--save-dir', type=str, default = '0')
    parser.add_argument('--show-vid', action = 'store_true')
    parser.add_argument('--save-vid', action = 'store_true')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
import cv2
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import os

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')
# files = [os.path.join(p,i) for i in sorted([i for i in os.listdir(p)], key=lambda i: int(i[:-4]))]
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        images = [i for i in sorted(images, key= lambda i:int(i.split('.')[-2].split('\\')[-1]))]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        gps = [i for i in files if '.txt' in i]
        gps = [i for i in sorted(gps, key = lambda i : int(i.split('.')[-2].split('\\')[-1]))]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.gps_files = gps
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    # def __iter__(self):
    #     self.count = 0
    #     return self
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        path = self.files[index]
        gps_info = pd.read_csv(self.gps_files[index])

        if self.video_flag[index]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[index]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            #print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        gps_info = gps_info.to_dict('list')

        return path, img, img0, gps_info

    # def __next__(self):
    #     if self.count == self.nf:
    #         raise StopIteration
    #     path = self.files[self.count]
    #     gps_info = pd.read_csv(self.gps_files[self.count])

    #     if self.video_flag[self.count]:
    #         # Read video
    #         self.mode = 'video'
    #         ret_val, img0 = self.cap.read()
    #         if not ret_val:
    #             self.count += 1
    #             self.cap.release()
    #             if self.count == self.nf:  # last video
    #                 raise StopIteration
    #             else:
    #                 path = self.files[self.count]
    #                 self.new_video(path)
    #                 ret_val, img0 = self.cap.read()

    #         self.frame += 1
    #         print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

    #     else:
    #         # Read image
    #         self.count += 1
    #         img0 = cv2.imread(path)  # BGR
    #         assert img0 is not None, 'Image Not Found ' + path
    #         #print(f'image {self.count}/{self.nf} {path}: ', end='')

    #     # Padded resize
    #     img = letterbox(img0, self.img_size, stride=self.stride)[0]

    #     # Convert
    #     img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    #     img = np.ascontiguousarray(img)
    #     gps_info = gps_info.to_dict()
    #     print(gps_info)
    #     print(type(gps_info))
    #     return path, img, img0, gps_info.to_dict(), self.cap
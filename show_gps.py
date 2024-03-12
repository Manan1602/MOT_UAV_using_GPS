import argparse
import torch
def run(yolo_weights,img_dir,img_size,show_vid=False,save_vid = False):
    model = torch.hub.load('WongKinYiu/yolov7','custom',yolo_weights,force_reload=True, trust_repo=True)
    print('done')


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
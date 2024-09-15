import argparse
from pathlib import Path
import time
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import csv
import os

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadOtherImages, LoadNorwayImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, TracedModel

from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader


class IteratorDataset(IterableDataset):
    def __init__(self, iterator):
        self.iterator = iterator
    
    def __iter__(self):
        return iter(self.iterator)

def custom_collate(batch):
    # Unzip the batch into separate lists
    path, img, im0s, origsize = zip(*batch)
    
    return path, np.array(img), im0s, origsize

def detect():
    source, weights, imgsz, trace, csv_file_path = opt.source, opt.weights, opt.img_size, not opt.no_trace, opt.csv_file_path
    
    # Load model
    # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(weights, map_location=device) # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    # import pdb
    # pdb.set_trace()
    if trace:
        if model.__class__.__name__ == "Ensemble":
            for i in range(len(model)):
                model[i] = TracedModel(model[i], device, imgsz)
        else:
            model = TracedModel(model, device, imgsz)

    if half:
        model.half()  # to FP16v

    with open(csv_file_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        predict(model, stride, imgsz, device, half, source, csv_writer, type='other')
        predict(model, stride, imgsz, device, half, source, csv_writer, type='norway')
        

    print('Done')


def predict(model, stride, imgsz, device, half, source, csv_writer, type='other'):
    if type == 'norway':
        # Set Dataloader
        dataset = LoadNorwayImages(source, img_size=imgsz, stride=stride)
        dataloader = DataLoader(IteratorDataset(dataset), batch_size=12, shuffle=False, collate_fn = custom_collate)
    else:
        dataset = LoadOtherImages(source, img_size=imgsz, stride=stride)
        dataloader = DataLoader(IteratorDataset(dataset), batch_size=24, shuffle=False, collate_fn = custom_collate)

    if len(dataset) == 0:
        return
    
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    for path, img, im0s, origsize in dataloader:
        # pdb.set_trace()
        img = torch.from_numpy(img).to(device)
        
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            im0 = im0s[i]
            # Get the image filename
            image_name = os.path.basename(path[i])
            prediction_string = ''

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for x_min, y_min, x_max, y_max, conf, cls in reversed(det):
                    if type == 'norway':
                        # origsize is (h, w) of the original/cropped image (not the imgsz fed in prediction)
                        h_pad = origsize[i][0] - 1824
                        prediction_string += f"{int(cls)+1} {int(x_min)} {int(y_min) + h_pad} {int(x_max)} {int(y_max)+h_pad} "
                    else:
                        prediction_string += f"{int(cls)+1} {int(x_min)} {int(y_min)} {int(x_max)} {int(y_max)} "
            # Write the row to the CSV file
            csv_writer.writerow([image_name, prediction_string.strip()]) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--imgsznorway', type=int, help='inference size for norway (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--norwayconf_thres', type=float, help='object confidence threshold for big image')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--csv-file-path', type=str, help='path to csv output file')
    opt = parser.parse_args()
    #print(opt)
    

    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        
        start_time = time.time()
        detect()
        print(f'time {time.time()-start_time} s')

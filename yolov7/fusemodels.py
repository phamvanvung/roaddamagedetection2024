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


def detect():
    weights, output_file_path = opt.weights, opt.output_file_path
    device = select_device(opt.device)
    model = attempt_load(weights, map_location=device)

    # import pdb
    # pdb.set_trace()
    # if trace:
    #     if model.__class__.__name__ == "Ensemble":
    #         for i in range(len(model)):
    #             model[i] = TracedModel(model[i], device, imgsz)
    #     else:
    #         model = TracedModel(model, device, imgsz)

    model.half()  # to FP16
    
    model.fuse()
    torch.save(model, output_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--output-file-path', type=str, help='path to csv output file')
    opt = parser.parse_args()
    #print(opt)
    

    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        
        start_time = time.time()
        detect()
        print(f'time {time.time()-start_time} s')

import os
import torch
import sys
import subprocess
import sys

# Setup
HOME = os.getcwd()
torch.cuda.empty_cache()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parse arguments
if len(sys.argv) != 3:
    print("Usage: python inference_script.py <images_path> <output_csv>")
    sys.exit(1)

source_path = sys.argv[1]      # Path to the directory containing images for inference
source_path = os.path.abspath(source_path)
csv_file_path = sys.argv[2]  # output CSV file name including directory name
csv_file_path = os.path.abspath(csv_file_path)

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(stderr.decode())
        sys.exit(1)
    return stdout.decode()


if __name__ == "__main__":
    if os.path.isdir("yolov7") or os.path.isdir('../yolov7'):
        if not os.path.basename(os.getcwd()) == 'yolov7':
            os.chdir("yolov7")
        # cmd = f"python detectyolov7.py --device 0 --weights  weights/best2022train9deploy.pt --conf 0.39 --img-size 640 --source {source_path} --augment --iou-thres 1.0 --csv-file-path {csv_file_path} --no-trace" # --norwayconf 0.34 --imgsznorway 960
        cmd = f"python batchdetectsinglefornorway.py --device 0 --weights  weights/best2022train9deploy.pt weights/besttinydeploy.pt --img-size 640 --source {source_path} --augment --iou-thres 1.0 --csv-file-path {csv_file_path} --no-trace  --conf 0.45 --norwayconf 0.35"
        #cmd = f"python batchdetectsinglefornorway.py --device 0 --weights weights/besttinydeploy.pt weights/best2022train9deploy.pt --img-size 640 --source {source_path} --augment --no-trace --iou-thres 1.0 --csv-file-path {csv_file_path}  --conf 0.48 --norwayconf 0.23"
        run_command(cmd)

#pip install ultralytics==8.0.88
import argparse
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

from ultralytics import YOLO

model = YOLO('YT_WZ_predict.pt')  # load a custom model

# Track with the model
#results = model.track(source="https://youtu.be/HupM1Lpnp2s") 
results = model.track(source="https://youtu.be/HupM1Lpnp2s", show=True) 
#results = model.track(source="https://youtu.be/HupM1Lpnp2s", show=True, tracker="bytetrack.yaml") 


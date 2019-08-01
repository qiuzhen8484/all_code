import numpy as np
import pandas as pd
import os
import cv2
import torch
import cPickle as pkl
import random
import math
from skimage import segmentation
from models import *
from fpn import FPN18
from skimage import exposure
from ModelUNetTogether import UNet
import ConfigParser


def crop_up_down():
    pass
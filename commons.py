import cv2
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import imutils
import sys
import numpy
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
numpy.set_printoptions(threshold=sys.maxsize)

channels = "HSCMYb"

def bgr_to_cmyk(bgr):
    RGB_SCALE = 255
    CMYK_SCALE = 100
    b = bgr[0]
    g = bgr[1]
    r = bgr[2]
    if (r, g, b) == (0, 0, 0):
        return 0, 0, 0, CMYK_SCALE

    c = 1 - r / RGB_SCALE
    m = 1 - g / RGB_SCALE
    y = 1 - b / RGB_SCALE

    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    return [c, m, y, k]

def flatten(img):
    return np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2]))

def features(bgr, channels, filter_white=True):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    cmyk = np.apply_along_axis(bgr_to_cmyk, 2, bgr)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    
    hsv_flat = flatten(hsv)
    lab_flat = flatten(lab)
    cmyk_flat = flatten(cmyk)
    bgr_flat = flatten(bgr)
    
    f = np.concatenate((bgr_flat, hsv_flat, cmyk_flat, lab_flat),1)
    if filter_white:
        f = [x for x in f if not np.sum(x[:3] == [255,255,255]) == 3] #remove white pixels
    f = np.array(f)
    channels_map = {'B':0, 'G':1,'R':2,'H':3,'S':4,'V':5,'C':6,'M':7,'Y':8,'K':9,'L':10,'A':11,'b':12}
    channels = list(channels)
    channels = [channels_map[ch] for ch in channels]
    f = f[:,channels]
    return np.array(f)
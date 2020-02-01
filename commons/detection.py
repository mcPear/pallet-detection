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
import pickle
import matplotlib.pyplot as plt
numpy.set_printoptions(threshold=sys.maxsize)
from features_extraction import *
from i_o import *

def loadNB():
    filename = '../model/naive_bayes_model.sav'
    gnb = pickle.load(open(filename, 'rb'))
    return gnb

def loadBlurredNB():
    filename_blurred = '../model/blurred_naive_bayes_model.sav'
    gnb_blurred = pickle.load(open(filename_blurred, 'rb'))
    return gnb_blurred

def divisible(num, factor=3):
    return (num % factor) == 0

def classify(img, model, with_save):
    w,h,ch = img.shape
    f = features(img, channels, False)
    pred = model.predict(f)
    img_pred_rev = np.reshape(pred, (w,h,1))
    img_pred = np.logical_not(img_pred_rev)
    if with_save:
        save(img_pred, "artifacts/img_classified.jpg")
    return img_pred

def median_filter(img, with_save):
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(img,-1,kernel)
    img = cv2.threshold(img,0.5,1.0,cv2.THRESH_BINARY)[1]
    if with_save:
        save(img, "artifacts/img_median.jpg")
    return img

def detect_by_height(img, mask_height, max_row_ind, min_perc, mask_holes_loss):
    results=[]
    mask_height = mask_height
    mask_width = int(mask_height * 5.556)
    mask_size = mask_height * mask_width
    hole_height = int(mask_height * 0.694)
    hole_width = int(mask_height * 1.58)
    hole_size = hole_height * hole_width
    hole_1_x = int(mask_height * 0.694)
    hole_1_y = int(mask_height * 0.306)
    hole_2_x = int(mask_height * 3.281)
    hole_2_y = hole_1_y
    hole_loss_ratio = 2 if mask_holes_loss else 1
    inverse_img = np.logical_not(img)
    img_height, img_width = inverse_img.shape
    for index, _ in np.ndenumerate(inverse_img):
        x,y = index
        y+=max_row_ind
        if divisible(x) and divisible(y) and y+mask_height < img_height and x+mask_width < img_width:
            frame_mask = inverse_img[y:y+mask_height, x:x+mask_width]
            hole_1_y_ = y+hole_1_y
            hole_2_y_ = y+hole_2_y
            hole_1_x_ = x+hole_1_x
            hole_2_x_ = x+hole_2_x
            hole_1_mask = inverse_img[hole_1_y_:hole_1_y_+hole_height, hole_1_x_:hole_1_x_+hole_width]
            hole_2_mask = inverse_img[hole_2_y_:hole_2_y_+hole_height, hole_2_x_:hole_2_x_+hole_width]
            frame_mask_perc = np.sum(frame_mask) / mask_size
            hole_1_mask_perc = np.sum(hole_1_mask) / hole_size
            hole_2_mask_perc = np.sum(hole_2_mask) / hole_size
            perc = frame_mask_perc - hole_loss_ratio*hole_1_mask_perc - hole_loss_ratio*hole_2_mask_perc
            result=(frame_mask, perc, (x,y), mask_height)
            if perc >= min_perc:
                results.append(result)
    return results

def mean_of_result_centers(results):
    centers=[np.array(x[2]) for x in results]
    mean = tuple(np.mean(centers, axis=0))
    return mean

def resolve_redundancy(results):
    bins=dict()
    acceptance_thr = 50
    for result in results:
        (frame_mask, perc, center, mask_height) = result
        matching_bins = [bin_key for bin_key in bins.keys() if dist(center, bin_key)<acceptance_thr]
        if matching_bins:
            matching_bin=matching_bins[0]
            all_bin_results = bins.pop(matching_bin)+[result]
            mean_center=mean_of_result_centers(all_bin_results)
            bins[mean_center] = all_bin_results
        else:
            bins[center]=[result]
    results=[max(bin_results, key=lambda x: x[1]) for bin_results in bins.values()]
    return results

def get_optim_row(img, margin, visualize):
    hist = [1.0 - np.mean(row) for row in img]
    max_row_ind = np.argmax(hist)
    optim_row_ind = max(0,max_row_ind-margin)
    if visualize:
        rows = np.arange(len(hist))
        print("optim_row_ind",optim_row_ind)
        plt.bar(rows, hist)
        plt.show()
    return optim_row_ind

def detect(img, min_height, max_height, step, min_perc, dense_row_margin, mask_holes_loss):
    results=[]
    most_dense_row=get_optim_row(img, dense_row_margin, visualize=False)
    for mask_height in range(min_height, max_height, step):
        mask_height_results = detect_by_height(img, mask_height, most_dense_row, min_perc, mask_holes_loss)
        results.extend(mask_height_results)
    return resolve_redundancy(results)

def dist(a,b): #fixme copy from evaluation.ipynb
    return np.linalg.norm(np.array(a)-np.array(b))

def draw_pallets(img, results, with_save):
    color = (0,255,0)
    for result in results:
        (frame_mask, perc, (x,y), mask_height)=result 
        mask_width = int(mask_height * 5.556)
        img[y:y+mask_height, x:x+1]=color
        img[y:y+mask_height, x+mask_width-1:x+mask_width]=color
        img[y:y+1, x:x+mask_width]=color
        img[y+mask_height-1:y+mask_height, x:x+mask_width]=color
    if with_save:
        save(img, "artifacts/img_marked.jpg", False)
    return img

def calculate_centers(results):
    centers=[]
    for result in results:
        (frame_mask, perc, (x,y), mask_height)=result 
        print("Fit:", perc)
        mask_width = int(mask_height * 5.556)
        x = x + mask_width/2
        y = y + mask_height/2
        center = (x,y)
        centers.append(center)
    return centers


def opening(img, with_save, kernel):
    img = 1-img
    kernel = np.ones(kernel, np.uint8) 
    img_erosion = cv2.erode(img, kernel, iterations=1)
    if with_save:
        save(1-img_erosion, "artifacts/img_erosion.jpg")
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    if with_save:
        save(1-img_dilation, "artifacts/img_dilation.jpg")
    return 1-img_dilation

def closing(img, with_save, kernel):
    img = 1-img
    kernel = np.ones(kernel, np.uint8) 
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    if with_save:
        save(1-img_dilation, "artifacts/img_dilation_2.jpg")
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    if with_save:
        save(1-img_erosion, "artifacts/img_erosion_2.jpg")
    return 1-img_erosion

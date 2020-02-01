import numpy as np
import cv2

def get_marks(filepath):
    marks = []
    img = cv2.imread(filepath)
    validation_sum = 3*255 - 50
    for x,y in np.ndindex(img.shape[:2]):
        if np.sum(img[(x,y)]) > validation_sum:
            marks.append((y,x))
    print(marks)
    return marks

def get_all_marks(path):
    marks = dict()
    filenames = [f for f in sorted(listdir(path)) if isfile(join(path, f))]
    for i in tqdm(range(len(filenames))):
        filename = filenames[i]
        print(path, filename)
        marks[filename] = get_marks(path + filename)
    return marks

def save_test_markers():
    path = "../data/"
    marks = get_all_marks(path)
    np.save(path+"markers/test_markers", marks)

def dist(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))
    
def evaluate():
    pred = np.load("../run all/detected/centers/detected_centers.npy", allow_pickle=True).item()
    test = np.load("../data/markers/test_markers.npy", allow_pickle=True).item()
    filenames = test.keys()
    FPs = 0
    TPs = 0
    FNs = 0
    acceptance_thr = 15
    for filename in filenames:
        marks = test[filename]
        marks_count=len(marks)
        found_marks=set()
        centers = pred[filename]
        for center in centers:
            distances = [dist(center, x) for x in marks]
            if distances:
                #print(distances)
                min_distance_idx = np.argmin(distances)
                min_distance=distances[min_distance_idx]
                if min_distance <= acceptance_thr:
                    found_marks.add(min_distance_idx)
                else:
                    FPs += 1
        found_marks_count = len(found_marks)
        FNs += marks_count - found_marks_count
        TPs += found_marks_count

    F1 = 2*TPs/(2*TPs + FPs + FNs)
    print("\nEvaluation: TPs:",TPs, " FPs:", FPs, " FNs:", FNs, " F1:", F1)
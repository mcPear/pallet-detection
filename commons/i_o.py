import cv2

def save(img, file, binary=True):
    res = cv2.imwrite(file, img * 255 if binary else img)
    print("saved" if res else "save error", file)
    
    
def show(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
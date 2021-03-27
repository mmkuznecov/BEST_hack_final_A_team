import cv2
import numpy as np
from keras.models import load_model

# loading model
model = load_model('models/unet.h5')

# applying before model prediction

def preprocess(img):
    im = np.zeros((256, 256, 3), dtype=np.uint8)

    if img.shape[0] >= img.shape[1]:
        scale = img.shape[0] / 256
        new_width = int(img.shape[1] / scale)
        diff = (256 - new_width) // 2
        img = cv2.resize(img, (new_width, 256))

        im[:, diff:diff + new_width, :] = img
    else:
        scale = img.shape[1] / 256
        new_height = int(img.shape[0] / scale)
        diff = (256 - new_height) // 2
        img = cv2.resize(img, (256, new_height))

        im[diff:diff + new_height, :, :] = img
        
    return im

# applying after model prediction

def postprocess(img_ori, pred):
    THRESHOLD = 0.8

    h, w = img_ori.shape[:2]
    
    mask_ori = (pred.squeeze()[:, :, 1] > THRESHOLD).astype(np.uint8)
    max_size = max(h, w)
    result_mask = cv2.resize(mask_ori, dsize=(max_size, max_size))

    if h >= w:
        diff = (max_size - w) // 2
        if diff > 0:
            result_mask = result_mask[:, diff:-diff]
    else:
        diff = (max_size - h) // 2
        if diff > 0:
            result_mask = result_mask[diff:-diff, :]
        
    result_mask = cv2.resize(result_mask, dsize=(w, h))
    
    result_mask *= 255

    # smoothen edges
    result_mask = cv2.GaussianBlur(result_mask, ksize=(9, 9), sigmaX=5, sigmaY=5)
    
    return result_mask


def process_frame(frame):
    img_ori = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
    
    img = preprocess(frame)
    input_img = img.reshape((1, 256, 256, 3)).astype(np.float32) / 255.

    pred = model.predict(input_img)
    
    THRESHOLD = 0.8
    EROSION = 1
    
    mask = postprocess(img_ori, pred)
    
    converted_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    result_img = cv2.subtract(converted_mask, img_ori)
    result_img = cv2.subtract(converted_mask, result_img)
    
    gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray, 1, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(gray, contours, -1, 255, 3)

        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        max_contour_area = cv2.contourArea(c)

        for i in contours:
            if cv2.contourArea(i) < max_contour_area:
                x,y,w,h = cv2.boundingRect(i)
                cv2.rectangle(result_img,(x,y),(x+w,y+h),(0,0,0),-1)
                
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    return result_img


# writing new vid
def process_vid(vid_path):
    cap = cv2.VideoCapture(vid_path)


    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(vid_path.split('.')[0] + '_out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    while(True):
        ret, frame = cap.read()

        if ret == True: 
            # segmented processed frame
            seg_frame = process_frame(frame)
            out.write(seg_frame)

        else:
            break  

    cap.release()
    out.release()
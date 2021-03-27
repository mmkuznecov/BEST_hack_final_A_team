import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_plot(vid_path):

    cap = cv2.VideoCapture(vid_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ratio=[]
    while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            cntPixels = height*width
            cntNotBlack = cv2.countNonZero(gray)
            cntBlack=cntPixels-cntNotBlack
            print('black',cntBlack)
            print('notblack',cntNotBlack)
            ratio.append(cntNotBlack/cntBlack)

    mean=np.full((5,),1/5)
    cov_ratio=np.convolve(ratio,mean,mode='valid')

    f,ax=plt.subplots(1)
    ax.plot(ratio,label='Original')
    ax.plot(cov_ratio,label='After convolution')
    ax.legend(loc='lower left')
    plt.show()

        
        

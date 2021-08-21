import os
import numpy as np #1.19.5
import cv2 #4.5.3
import time
import handtrackingmodule as track

closed_image = cv2.imread('Images\Closed.PNG')
open_1 = cv2.imread('Images\open_1.PNG')
open_2 = cv2.imread('Images\open_2.PNG')
open_3 = cv2.imread('Images\open_3.PNG')
open_4 = cv2.imread('Images\open_4.PNG')
open_5 = cv2.imread('Images\open_5.PNG')

width,height = 640,480

detector = track.HandDetector(detectioncon=0.65)
cap_vid = cv2.VideoCapture(0)
cap_vid.set(3,width)
cap_vid.set(4,height)

ptime = 0

tipid = [4,8,12,16,20]

while True:
    success, img = cap_vid.read()

    img = detector.findhands(img)
    lmlist = detector.findposition(img,id_to_track=1,draw=True)
    # print(lmlist)
    if lmlist:
        if lmlist[tipid[0]][1] < lmlist[tipid[4]][1]:
            fingers=[]
            if lmlist[tipid[0]][1] < lmlist[tipid[0]-1][1]:
                fingers.append(1) 
            else:
                fingers.append(0)
            for id in range(1,5):
                if lmlist[tipid[id]][2] < lmlist[tipid[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # print(fingers)
            if fingers == [0,0,0,0,0]:
                img[0:closed_image.shape[0],0:closed_image.shape[1]] = closed_image
            elif fingers == [0,1,0,0,0]:
                img[0:open_1.shape[0],0:open_1.shape[1]] = open_1
            elif fingers == [0,1,1,0,0]:
                img[0:open_2.shape[0],0:open_2.shape[1]] = open_2
            elif fingers == [0,1,1,1,0]:
                img[0:open_3.shape[0],0:open_3.shape[1]] = open_3
            elif fingers == [0,1,1,1,1]:
                img[0:open_4.shape[0],0:open_4.shape[1]] = open_4
            elif fingers == [1,1,1,1,1]:
                img[0:open_5.shape[0],0:open_5.shape[1]] = open_5
        
        elif lmlist[tipid[0]][1] > lmlist[tipid[4]][1]:
            fingers=[]
            if lmlist[tipid[0]][1] > lmlist[tipid[0]-1][1]:
                fingers.append(1) 
            else:
                fingers.append(0)
            for id in range(1,5):
                if lmlist[tipid[id]][2] < lmlist[tipid[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # print(fingers)
            if fingers == [0,0,0,0,0]:
                img[0:closed_image.shape[0],0:closed_image.shape[1]] = closed_image
            elif fingers == [0,1,0,0,0]:
                img[0:open_1.shape[0],0:open_1.shape[1]] = open_1
            elif fingers == [0,1,1,0,0]:
                img[0:open_2.shape[0],0:open_2.shape[1]] = open_2
            elif fingers == [0,1,1,1,0]:
                img[0:open_3.shape[0],0:open_3.shape[1]] = open_3
            elif fingers == [0,1,1,1,1]:
                img[0:open_4.shape[0],0:open_4.shape[1]] = open_4
            elif fingers == [1,1,1,1,1]:
                img[0:open_5.shape[0],0:open_5.shape[1]] = open_5
    
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(img,f'FPS:{int(fps)}',(514,30),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)



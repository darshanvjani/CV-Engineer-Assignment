import cv2
import mediapipe as mp #0.8.7
import time

from mediapipe.python.solutions.hands import HandLandmark

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectioncon=0.5, trackingcon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectioncon = detectioncon
        self.trackingcon = trackingcon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectioncon,self.trackingcon)
        self.mpDraw = mp.solutions.drawing_utils


    def findhands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for landmarks in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,landmarks,self.mpHands.HAND_CONNECTIONS)

        return img

    def findposition(self,img,handNo=0,id_to_track=None,draw=False):

        lmList = []
        if self.result.multi_hand_landmarks:
            myhand = self.result.multi_hand_landmarks[handNo]

            for id_,lm in enumerate(myhand.landmark):
                 h,w,c = img.shape
                 cw,ch = int(lm.x*w) , int(lm.y*h)
                 lmList.append([id_,cw,ch])

                 if draw:
                    if id_to_track == id_:
                        cv2.circle(img,(cw,ch),15,(255,0,255),cv2.FILLED)
                    if id_to_track == None:
                        cv2.circle(img,(cw,ch),10,(0,0,255),cv2.FILLED)

        return lmList



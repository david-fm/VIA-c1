#!/usr/bin/env python


import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText


import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands()

for _, frame in autoStream():
    H,W, _ = frame.shape
    imagecv = cv.flip(frame,1)
    image = cv.cvtColor(imagecv,cv.COLOR_BGR2RGB)
    
    results = hands.process(image)
  
    black_image = np.zeros(image.shape, dtype=np.uint8)
    # Image used to train the model
    points = []

    if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            for k in range(21):
              """https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
              información de para que sirve cada punto de la mano
              """
              x = hand_landmarks.landmark[k].x # entre 0 y 1
              y = hand_landmarks.landmark[k].y
              points.append([int(x*W), int(y*H)]) # int para dibujar en cv
            break

          points = np.array(points) 
          # mejor un array para poder operar matematicamente
          print(points)

          # dibujar un segmento de recta en el dedo índice
          # cv.line(imagecv, points[5], points[8], color=(0,255,255), thickness = 3)

          # center = np.mean(points[[5,0,17]], axis=0)
          # radio =  np.linalg.norm(center -points[5])
          # cv.circle(imagecv, center.astype(int), int(radio), color=(0,255,255), thickness = 3)

          mp_drawing.draw_landmarks(
                imagecv,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
          
          mp_drawing.draw_landmarks(
                black_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    #cv.imshow("manos", imagecv)
    cv.imshow("manos fondo negro", black_image)
    # cv.imshow("mirror", image)


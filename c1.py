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

center = np.array([0,0]) # punto que se dibujará según la posición de la mano
angle = 0 # ángulo de la mano
percentage = 0 # porcentaje de ocupación de la mano
MAX_POINT_RADIUS = 50 # tamaño máximo del punto
closed = False # True si la mano está cerrada, False si no


def hand_angle(point_0, point_17):
    # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
    # 0: wrist, 17: pinky_mcp

    point_17 = point_17 - point_0

    
    angle = np.arctan2(point_17[1], point_17[0]) * 180 / np.pi
    return angle

def percentage_ocupied(points, W, H):
    # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
    # 0: wrist, 17: pinky_mcp, 5: index_finger_mcp

    center = np.mean(points[[5,0,17]], axis=0)
    radio =  center -points[0]

    module = np.linalg.norm(radio)

    # print(f"module: {module}")
    # print(f"center: {center}")
    # print(f"points[0]: {points[0]}")
    # print(f"radio: {radio}")
    # print(f"W/2: {H/2}")
    #print(f"my module: {np.sqrt(moved_radio[0]**2 + moved_radio[1]**2)}")

    return module/(H/2)

def hand_closed(points, center):
    """Returns True if the hand is closed, False otherwise"""
    # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
    # 4: thumb_tip, 8: index_tip, 12: middle_tip, 16: ring_tip, 20: pinky_tip

    CONSTANTE = 250

    distance_4_center = np.linalg.norm(points[4] - center)
    distance_8_center = np.linalg.norm(points[8] - center)
    distance_12_center = np.linalg.norm(points[12] - center)
    distance_16_center = np.linalg.norm(points[16] - center)
    distance_20_center = np.linalg.norm(points[20] - center)

    sum_distances = distance_4_center + distance_8_center + distance_12_center + distance_16_center + distance_20_center

    return sum_distances < CONSTANTE

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
          #print(points)
          center = np.mean(points, axis=0).astype(int)
          angle = hand_angle(points[0], points[17])
          angle = angle if angle > 0 else -angle
          percentage = percentage_ocupied(points, W, H)
          closed = hand_closed(points, center)

          # dibujar un segmento de recta en el dedo índice
          # cv.line(imagecv, points[5], points[8], color=(0,255,255), thickness = 3)

          # center = np.mean(points[[5,0,17]], axis=0)
          # radio =  np.linalg.norm(center -points[5])
          # cv.circle(imagecv, center.astype(int), int(radio), color=(0,255,255), thickness = 3)

          # mp_drawing.draw_landmarks(
          #       imagecv,
          #       hand_landmarks,
          #       mp_hands.HAND_CONNECTIONS,
          #       mp_drawing_styles.get_default_hand_landmarks_style(),
          #       mp_drawing_styles.get_default_hand_connections_style())
          
          mp_drawing.draw_landmarks(
                black_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
  
    #cv.imshow("manos", imagecv)

    blue = int((1-angle/180)*255)
    red = int((angle/180)*255)
    cv.putText(black_image, f"blue: {blue}, red: {red}, angle: {angle:.2f}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)
    cv.putText(black_image, f"x: {center[0]}, y: {center[1]}", (10,60), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)
    cv.putText(black_image, f"Hand closed: {closed}", (10,90), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)
    cv.circle(black_image, center, int(MAX_POINT_RADIUS*percentage), (blue,0,red), -1)
    cv.imshow("manos fondo negro", black_image)
    # cv.imshow("mirror", image)


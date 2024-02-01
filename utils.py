import os
import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle

PATH = os.path.join(os.path.dirname(__file__), 'data')
RESULT_PATH = os.path.join(os.path.dirname(__file__), 'results')
LETTERS = ["a", "e", "i", "o", "u"]

def get_results(images):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands()
    results = []
    all_points = []
    
    for image in images:
        H,W, _ = image.shape
        black_image = np.zeros(image.shape, dtype=np.uint8)
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        hand_results = hands.process(image)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                points = []
                for k in range(21):
                    """https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
                    información de para que sirve cada punto de la mano
                    """
                    x = hand_landmarks.landmark[k].x # entre 0 y 1
                    y = hand_landmarks.landmark[k].y
                    points.append([int(x*W), int(y*H)]) # int para dibujar en cv
                    break

                points = np.array(points)
                all_points.append(points)
                mp_drawing.draw_landmarks(
                        black_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                results.append(black_image)
    return results, all_points



def save_images(images, folder):
    for i, image in enumerate(images):
        cv.imwrite(os.path.join(folder, f"{i}.png"), image)

def save_points(points, folder):
    for i, points in enumerate(points):

        with open(os.path.join(folder, f"points{i}.pkl"), "ab") as f:
            pickle.dump(points, f)

def load_images(folder):
    images = []
    for file in os.listdir(folder):
        file = os.path.join(folder,file)
        img = cv.imread(file)
        images.append(img)
    return images

if __name__ == "__main__":

    for letter in LETTERS:
        images = load_images(os.path.join(PATH, letter))
        results, all_points = get_results(images)
        save_images(results, os.path.join(RESULT_PATH, letter))
        save_points(all_points, os.path.join(RESULT_PATH, letter))


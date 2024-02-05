import numpy as np
import mediapipe as mp
import cv2 as cv
import os
import pickle
from umucv.stream import autoStream



class HandDetector():
    def __init__(self, folder):
        self.points = {
            "a": [],
            "e": [],
            "i": [],
            "o": [],
            "u": []
        }
        # model is a dictionary with the key of the letter and a base example of the hand
        # Importante, normalizar distancias en base a la distancia entre el punto 0 y el punto 9
        # 0 es la muñeca y 9 es la base del dedo medio
        self.model: dict[str,list(list(int))] = {
            "a": None,
            "e": None,
            "i": None,
            "o": None,
            "u": None
        }

        for subfolder in os.listdir(folder):
            i = 0
            for file in os.listdir(os.path.join(folder,subfolder)):
                if file.endswith(".png"):
                    continue
                
                file = os.path.join(folder,subfolder,file)
                with open(file, "rb") as f:
                    points = pickle.load(f)


                    if i == 0:
                        self.model[subfolder] = points
                    else:
                        self.points[subfolder].append(points)
                    i += 1
   
    @staticmethod
    def transform( reference, to_transform):
        """Transform the points of to_transform to the reference"""
        reference = np.array(reference)
        to_transform = np.array(to_transform)
        movement = reference[0] - to_transform[0]
        # Translate the points to the origin
        to_transform = to_transform + movement
        HandDetector.test(to_transform)
        # Rotate the points based on the point 9
        # Change the origin to the point 9
        # TODO Improve
        point9 = to_transform[9]
        to_transform = to_transform - point9
        reference = reference - point9
        angle = np.arctan2(reference[9][1], reference[9][0]) - np.arctan2(to_transform[9][1], to_transform[9][0]) # angle in radians arctan2(y,x)
        to_transform = to_transform @ np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]]) # REVISAR

        # Set the origin back to the original point
        to_transform = to_transform + point9
        #HandDetector.test(to_transform, "rotated")
        return to_transform
    @staticmethod
    def test(transformed, name="test"):
        
        image = np.zeros((480,640,3), dtype=np.uint8)
        # transform to int
        t = transformed
        transformed = transformed.astype(int)
        for i in range(21):
            cv.circle(image, tuple(transformed[i]), 5, (0,255,255), -1)

        for _, frame in autoStream():

            cv.imshow(name, image)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break


    def train(self):
        pass
    

    def predict(self, image):
        return self.model.predict(image)


if __name__ == "__main__":
    PREPROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), 'results')

    detector = HandDetector(PREPROCESSED_DATA_PATH)
    #transformed = detector.transform(detector.model["a"], detector.points["a"][0])
    #detector.test(transformed)
    a = detector.model["a"]
    scale = np.linalg.norm(a[0] - a[9])/np.linalg.norm(a[0] - a[9])
    a = a*scale
    detector.test(a)
    b=detector.points["a"][5]
    detector.test(b)
    b = detector.transform(a,b)
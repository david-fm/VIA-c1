import numpy as np
import matplotlib.pyplot as plt
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
        #HandDetector.test(to_transform, "translated")
        # Rotate the points based on the point 9
        # Change the origin to the point 9
        # TODO Improve
        point0 = to_transform[0]
        to_transform = to_transform - point0
        reference = reference - point0
        angle = np.arctan2(reference[9][1], reference[9][0]) - np.arctan2(to_transform[9][1], to_transform[9][0]) # angle in radians arctan2(y,x)
        cos = np.cos(angle)
        sin = np.sin(angle)
        transformation_matrix = np.array([[cos, -sin],[sin, cos]])
        to_transform = to_transform @ transformation_matrix
        #print(f"reference angle: {np.arctan2(reference[9][1], reference[9][0])}")
        #print(f"to_transform angle: {np.arctan2(to_transform[9][1], to_transform[9][0])}")
        #print(f"angle: {angle}, cos: {cos}, sin: {sin}")
        # Set the origin back to the original point
        
        #HandDetector.test(to_transform, "rotated")
        to_transform = to_transform + point0
        #HandDetector.test(to_transform, "transformed back origin")
        return to_transform
    @staticmethod
    def plot(transformed, name="test"):
        fig, ax = plt.subplots()
        # transform to int
        for i in range(21):
            x, y = transformed[i]
            ax.plot(x, y)
            ax.text(x, y, str(i))

        conections = [
            [0,1,2,3,4],
            [0,5,6,7,8],
            [9,10,11,12],
            [13,14,15,16],
            [0,17,18,19,20],
            [5,9,13,17]
        ]
        # draw lines between points in each array of conections
        for conections in conections:
            for i in range(len(conections) - 1):
                x = [transformed[conections[i]][0], transformed[conections[i+1]][0]]
                y = [transformed[conections[i]][1], transformed[conections[i+1]][1]]
                ax.plot(x,y)
        
        reference = (1,0)
        ax.plot(reference[0], reference[1], "o")

        plt.title(name)
        plt.show()


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
    detector.plot(a, "base")
    b=detector.points["a"][0]
    b = detector.transform(a,b)
    detector.plot(b, "transformed")
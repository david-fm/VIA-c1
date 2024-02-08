import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# TODO: No tener un solor modelo de referencia sino un ensamble model con multiples ejemplos

# TODO: Marimo para ejemplificar

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
                        self.model[subfolder] = HandDetector.scale(points - points[0])
                    else:
                        self.points[subfolder].append(points)
                    i += 1
   
    @staticmethod
    def scale( points):
        # scale the points based on the distance between the point 0 and the point 9
        distance09 = abs(points[9] - points[0])
        scaled_points = points/distance09
        return scaled_points

    @staticmethod
    def transform( reference, to_transform):
        """Transform the points of to_transform to the reference"""
        
        reference = np.array(reference)
        to_transform = np.array(to_transform)
        # Translate the points to the same position as the reference
        # Given that the model reference is has the point 0 in the origin
        # the translation will make the point 0 of the to_transform to be at the origin
        movement = reference[0] - to_transform[0]
        
        to_transform = to_transform + movement


        # IMPORTANT: ROTATION IS NOT NECESSARY AFTER SCALING THE POINTS

        # angle = np.arctan2(reference[9][1], reference[9][0]) - np.arctan2(to_transform[9][1], to_transform[9][0]) # angle in radians arctan2(y,x)
        # cos = np.cos(angle)
        # sin = np.sin(angle)
        # transformation_matrix = np.array([[cos, -sin],[sin, cos]])
        # to_transform = to_transform @ transformation_matrix
        
        #fig, ax = HandDetector.plot(to_transform, "transformed", fig, ax, "g")
        to_transform = HandDetector.scale(to_transform)
        return to_transform


    @staticmethod
    def plot(transform, name="test", fig=None, ax=None, color=None):
        """Plot the transformed points, allow to plot multiple points in the same figure"""
        if fig is None:
            fig, ax = plt.subplots()
        # transform to int
        for i in range(21):
            x, y = transform[i]
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
                x = [transform[conections[i]][0], transform[conections[i+1]][0]]
                y = [transform[conections[i]][1], transform[conections[i+1]][1]]
                if color is not None:
                    ax.plot(x,y, color=color)
                else:
                    ax.plot(x,y)
        
        reference = (1,0)
        if color is not None:
            ax.plot(reference[0], reference[1], "o", color=color)
        else:
            ax.plot(reference[0], reference[1], "o")

        ax.set_title(name)
        return fig, ax

    @staticmethod
    def distance( a, b):
        return a-b
    
    @staticmethod
    def distanceMean(distance: list[list[int]]):
        return np.linalg.norm( distance, axis=1).mean()
    
    def distanceMedian(distance: list[list[int]]):
        return np.median(np.linalg.norm( distance, axis=1))

    def predict(self, points: list[list[int]], metric="mean"):
        # Search for the most similar after the transformation
        # Get the transformed points
        transformedA = self.transform(self.model["a"], points)
        transformedE = self.transform(self.model["e"], points)
        transformedI = self.transform(self.model["i"], points)
        transformedO = self.transform(self.model["o"], points)
        transformedU = self.transform(self.model["u"], points)

        # Calculate the distance between the transformed points and the model points
        distanceA = HandDetector.distance(transformedA, self.model["a"])
        distanceE = HandDetector.distance(transformedE, self.model["e"])
        distanceI = HandDetector.distance(transformedI, self.model["i"])
        distanceO = HandDetector.distance(transformedO, self.model["o"])
        distanceU = HandDetector.distance(transformedU, self.model["u"])
        letters = ["a", "e", "i", "o", "u"] 
        distances = [distanceA, distanceE, distanceI, distanceO, distanceU]

        measures = []
        for distance in distances:
            if metric == "mean":
                measures.append(HandDetector.distanceMean(distance))
            else:
                measures.append(HandDetector.distanceMedian(distance))
        
        return letters[measures.index(min(measures))]

        
    def accuracy(self, metric="mean"):
        # Measure how well the model is working
        accuracy = 0
        num_examples = 0
        for letter in self.points.keys():
            num_examples += len(self.points[letter])
            for point in self.points[letter]:
                if self.predict(point, metric=metric) == letter:
                    accuracy += 1
        
        return accuracy/num_examples
        



if __name__ == "__main__":
    PREPROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), 'results')

    detector = HandDetector(PREPROCESSED_DATA_PATH)
    
    print(f"Acuracy with median: {detector.accuracy(metric='median')}")
    print(f"Acuracy with mean: {detector.accuracy(metric='mean')}")
    

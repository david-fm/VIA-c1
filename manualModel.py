import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import utils

# TODO: No tener un solor modelo de referencia sino un ensamble model con multiples ejemplos

# TODO: Marimo para ejemplificar

class HandDetector():
    def __init__(self):
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

        self.model = {
            "a": [],
            "e": [],
            "i": [],
            "o": [],
            "u": []
        }
    
    def loadModelPoints(self, folder):
        """
            Load points from a folder with the points organized in subfoders
            Use this method to test the model accuracy
        """

        for subfolder in os.listdir(folder):
            images = utils.load_images(os.path.join(folder,subfolder))
            _, points = utils.get_results(images)
            for hand_points in points:
                self.model[subfolder].append(HandDetector.scale(hand_points - hand_points[0]))

    def loadPoints(self, folder):
        """
            Load points from a folder with the points organized in subfoders
            Use this method to test the model accuracy
        """

        for subfolder in os.listdir(folder):
            images = utils.load_images(os.path.join(folder,subfolder))
            _, points = utils.get_results(images)
            for hand_points in points:
                    self.points[subfolder].append(hand_points)
    
    @staticmethod
    def getPointsFromImage(image):
        """
            Get the points from the image
            Parameters:
                image: image path
            Returns:
                points: list of points
        """
        image = cv2.imread(image)
        _, points = utils.get_results([image])
        return points[0]
    
    def classifyImage(self, image, verbose=False):
        """
            Classify the image
            Parameters:
                image: image path
            Returns:
                letter: letter of the image
        """
        points = HandDetector.getPointsFromImage(image)
        return self.predict(points, verbose=verbose)
    
    @staticmethod
    def scale( points):
        # scale the points based on the distance between the point 0 and the point 9
        distance09 = abs(points[9] - points[0])
        # change distance09 values that are 0 to 1 to avoid division by 0
        distance09 = np.where(distance09 == 0, 1, distance09)
        scaled_points = points/distance09
        return scaled_points

    @staticmethod
    def transform( references, to_transform):
        """Transform the points of to_transform to the reference"""
        transformed_points = []
        for reference in references:
            reference = np.array(reference)
            transforming = np.array(to_transform)
            # Translate the points to the same position as the reference
            # Given that the model reference is has the point 0 in the origin
            # the translation will make the point 0 of the to_transform to be at the origin
            movement = reference[0] - transforming[0]
            
            transforming = transforming + movement


            # IMPORTANT: ROTATION IS NOT NECESSARY AFTER SCALING THE POINTS

            # angle = np.arctan2(reference[9][1], reference[9][0]) - np.arctan2(to_transform[9][1], to_transform[9][0]) # angle in radians arctan2(y,x)
            # cos = np.cos(angle)
            # sin = np.sin(angle)
            # transformation_matrix = np.array([[cos, -sin],[sin, cos]])
            # to_transform = to_transform @ transformation_matrix
            
            #fig, ax = HandDetector.plot(to_transform, "transformed", fig, ax, "g")
            transforming = HandDetector.scale(transforming)
            transformed_points.append(transforming)
        return transformed_points


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

    def predict(self, points: list[list[int]], metric="mean", verbose=False):
        # Search for the most similar after the transformation
        # Get the transformed points
        transformedA = self.transform(self.model["a"], points)
        transformedE = self.transform(self.model["e"], points)
        transformedI = self.transform(self.model["i"], points)
        transformedO = self.transform(self.model["o"], points)
        transformedU = self.transform(self.model["u"], points)

        # Calculate the distance between the transformed points and the model points

        distancesA = HandDetector.distance(np.array(transformedA), np.array(self.model["a"]))
        distancesE = HandDetector.distance(np.array(transformedE), np.array(self.model["e"]))
        distancesI = HandDetector.distance(np.array(transformedI), np.array(self.model["i"]))
        distancesO = HandDetector.distance(np.array(transformedO), np.array(self.model["o"]))
        distancesU = HandDetector.distance(np.array(transformedU), np.array(self.model["u"]))
        letters = ["a", "e", "i", "o", "u"] 
        distances = [distancesA, distancesE, distancesI, distancesO, distancesU]

        measures = []
        for distance in distances:
            letter_measures = []
            for case in distance:
                if metric == "mean":
                    letter_measures.append(HandDetector.distanceMean(case))
                else:
                    letter_measures.append(HandDetector.distanceMedian(case))
            # Append the best measure for each letter
            measures.append(min(letter_measures))

        
        if verbose:
            return letters[measures.index(min(measures))], measures

        return letters[measures.index(min(measures))]
        
    def accuracy(self, metric="mean"):
        # Measure how well the model is working
        accuracy = 0
        num_examples = 0
        for letter in self.points.keys():
            num_examples += len(self.points[letter])
            for points_example in self.points[letter]:
                if self.predict(points_example, metric=metric) == letter:
                    accuracy += 1
        
        return accuracy/num_examples
        



if __name__ == "__main__":
    MODELS_PATH = os.path.join(os.path.dirname(__file__), 'models')
    A_EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), 'data','a','20240201-185821.png')

    hd = HandDetector()
    
    hd.loadModelPoints(MODELS_PATH)
    points = hd.getPointsFromImage(A_EXAMPLE_PATH)
    hd.predict(points, verbose=True)

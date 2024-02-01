import tensorflow as tf
import os
import cv2 as cv
from abc import ABC, abstractmethod

from keras import  layers, models
import pickle

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
PREPROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), 'results')
TEST_COMBINATIONS = 5

MAP_LETTER = {
            "a": 0,
            "e": 1,
            "i": 2,
            "o": 3,
            "u": 4
        }

class Model(ABC):
    @abstractmethod
    def train(self, train_images, train_labels, epochs=10):
        pass
    @abstractmethod
    def save(self, name='my_model'):
        pass
    @abstractmethod
    def load(self, name='my_model'):
        pass
    @abstractmethod
    def predict(self, image):
        pass
    @abstractmethod
    def evaluate(self, test_images, test_labels):
        pass

# Based on https://www.tensorflow.org/tutorials/images/cnn?hl=es-419
class LinearModel(Model):
    def __init__(self, hidden_units = 64):
        """Linear model with 5 outputs and 21 inputs"""
        self.model = models.Sequential()
        self.model.add(layers.Input(shape=(21,)))
        if hidden_units != 0:
            self.model.add(layers.Dense(hidden_units, activation='relu'))
        self.model.add(layers.Dense(5))
        self.model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    
    @staticmethod
    def get_data(folder):
        """given a folder with 5 folders (a,e,i,o,u) return train_points, train_labels, test_points, test_labels"""
        train_points = []
        train_labels = []
        test_points = []
        test_labels = []
        

        for subfolder in os.listdir(folder):
            len_folder = len(os.listdir(os.path.join(folder,subfolder)))
            # take 10% of the images for testing
            for i,file in enumerate(os.listdir(os.path.join(folder,subfolder))):
                if file.endswith(".png"):
                    continue
                
                file = os.path.join(folder,subfolder,file)
                with open(file, "rb") as f:
                    points = pickle.load(f)
                    if i < len_folder*0.1:
                        test_points.append(points)
                        test_labels.append(subfolder)
                    else:
                        train_points.append(points)
                        train_labels.append(subfolder)
        
        train_labels = [MAP_LETTER[label] for label in train_labels]
        train_points = tf.convert_to_tensor(train_points)
        train_labels = tf.convert_to_tensor(train_labels)
        test_labels = [MAP_LETTER[label] for label in test_labels]
        test_labels = tf.convert_to_tensor(test_labels)
        test_points = tf.convert_to_tensor(test_points)
        
        return train_points, train_labels, test_points, test_labels

    def train(self, train_images, train_labels, epochs=10):
        self.model.fit(train_images, train_labels, epochs=epochs, shuffle=True)
    def save(self, name='my_model'):
        self.model.save(name)
    def load(self, name='my_model'):
        self.model = models.load_model(name)
    def predict(self, image):
        return self.model.predict(image)
    def evaluate(self, test_images, test_labels):
        test_loss, test_acc = self.model.evaluate(test_images,  test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)
        print('\nTest loss:', test_loss)
        return (test_loss,test_acc)
        

class ConvModel(Model):
    def __init__(self, conv_layer_filters = [32, 64, 64], dense_layer_units = 64):

        if len(conv_layer_filters) != 3:
            raise ValueError("conv_layer_filters must have 3 elements")
        if conv_layer_filters[0] == 0:
            raise ValueError("conv_layer_filters[0] must be greater than 0")
        

        filter1, filter2, filter3 = conv_layer_filters
        dense1 = dense_layer_units

        if filter1 < 0 or filter2 < 0 or filter3 < 0 or dense1 < 0:
            raise ValueError("All values must be greater than 0")

        self.model = models.Sequential()
        
        self.model.add(layers.Conv2D(filter1, (3, 3), activation='relu', input_shape=(480, 640, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        if filter2 != 0:
            self.model.add(layers.Conv2D(filter2, (3, 3), activation='relu'))
            self.model.add(layers.MaxPooling2D((2, 2)))
        if filter3 != 0:
            self.model.add(layers.Conv2D(filter3, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())

        if dense1 != 0:
            self.model.add(layers.Dense(dense1, activation='relu'))
        
        self.model.add(layers.Dense(5))

        self.model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        
        self.model.fit
    
    @staticmethod
    def get_data(folder):
        """given a folder with 5 folders (a,e,i,o,u) return train_images and train_labels"""
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []
        

        for f in os.listdir(folder):
            len_folder = len(os.listdir(os.path.join(folder,f)))
            # take 10% of the images for testing
            for i,file in enumerate(os.listdir(os.path.join(folder,f))):
                if file.endswith(".pkl"):
                    continue
                file = os.path.join(folder,f,file)
                img = cv.imread(file)
                if i < len_folder*0.1:
                    test_images.append(img)
                    test_labels.append(f)
                else:
                    train_images.append(img)
                    train_labels.append(f)
        
        train_labels = [MAP_LETTER[label] for label in train_labels]
        train_images = tf.convert_to_tensor(train_images)
        train_images = train_images/255
        train_labels = tf.convert_to_tensor(train_labels)
        test_labels = [MAP_LETTER[label] for label in test_labels]
        test_labels = tf.convert_to_tensor(test_labels)
        test_images = tf.convert_to_tensor(test_images)
        test_images = test_images/255
        
        return train_images, train_labels, test_images, test_labels
    
    def train(self, train_images, train_labels, epochs=10):
        self.model.fit(train_images, train_labels, epochs=epochs, shuffle=True)
    
    def save(self, name='my_model'):
        self.model.save(name)
    
    def load(self, name='my_model'):
        self.model = models.load_model(name)

    def predict(self, image):
        return self.model.predict(image)
    
    def evaluate(self, test_images, test_labels):
        test_loss, test_acc = self.model.evaluate(test_images,  test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)
        print('\nTest loss:', test_loss)
        return (test_loss,test_acc) 

class ConvTrainingGrid:
    def __init__(self, conv_layer_filters: [[int]], dense_layer_units: [int], epochs: [int]):
        self.conv_layer_filters = conv_layer_filters
        self.dense_layer_units = dense_layer_units
        self.epochs = epochs

    def train(self, train_images, train_labels) -> [(str,ConvModel)]:
        models = []
        for c in self.conv_layer_filters:
            for d in self.dense_layer_units:
                for e in self.epochs:
                    # Create 10 models with the same parameters and train them
                    # then return all of them to future comparison
                    combinations = []
                    for _ in range(TEST_COMBINATIONS):
                        model = ConvModel(c, d)
                        model.train(train_images, train_labels, e)
                        combinations.append(model)
                    # append a tupple with a string representing the combination of parameters and the model
                    models.append((f"convolutional layers: {c}; dense layer: {d} epochs: {e}", combinations))
        return models
    
    def compare(self, models: [(str,ConvModel)], test_images, test_labels):
        results = []
        for m in models:
            best_from_combination = None
            mean_accuracy = 0
            mean_loss = 0
            for model in m[1]:
                loss, acc = model.evaluate(test_images, test_labels)
                mean_accuracy += acc
                mean_loss += loss
                if best_from_combination is None or acc > best_from_combination[1]:
                    best_from_combination = model
            mean_accuracy /= TEST_COMBINATIONS
            mean_loss /= TEST_COMBINATIONS

            results.append((m[0], (mean_loss,mean_accuracy), best_from_combination))
        for r in results:
            print(r[0],"test loss: "+str(r[1][0])+" test accuracy: "+str(r[1][0]))
        
        # return the best model
        return max(results, key=lambda x: x[1])[2]

class LinearTrainingGrid:
    def __init__(self, dense_layer_units: [int], epochs: [int]):
        self.dense_layer_units = dense_layer_units
        self.epochs = epochs

    def train(self, train_points, train_labels) -> [(str,LinearModel)]:
        models = []
        for d in self.dense_layer_units:
            for e in self.epochs:
                # Create 10 models with the same parameters and train them
                # then return all of them to future comparison
                combinations = []
                for _ in range(TEST_COMBINATIONS):
                    model = LinearModel(d)
                    model.train(train_points, train_labels, e)
                    combinations.append(model)
                # append a tupple with a string representing the combination of parameters and the model
                models.append((f"dense layer: {d} epochs: {e}", combinations))
        return models
    
    def compare(self, models: [(str,LinearModel)], test_points, test_labels):
        results = []
        for m in models:
            best_from_combination = None
            mean_accuracy = 0
            mean_loss = 0
            for model in m[1]:
                loss, acc = model.evaluate(test_points, test_labels)
                mean_accuracy += acc
                mean_loss += loss
                if best_from_combination is None or acc > best_from_combination[1]:
                    best_from_combination = model
            mean_accuracy /= TEST_COMBINATIONS
            mean_loss /= TEST_COMBINATIONS

            results.append((m[0], (mean_loss,mean_accuracy), best_from_combination))
        for r in results:
            print(r[0],"test loss: "+str(r[1][0])+" test accuracy: "+str(r[1][0]))
        
        # return the best model
        return max(results, key=lambda x: x[1])[2]

if __name__ == '__main__':
    train_grid_conv = [
        # Convolutional layers
        [[32, 64, 64], [32, 64, 0], [32, 0, 0], [10, 20, 20], [10,20,0]],
        #Dense layers
        [0, 10, 20, 64],
        # Epochs
        [5,10, 15, 20]]
    mini_train_grid_conv = [
        # Convolutional layers
        [[32, 64, 64]],
        #Dense layers
        [64],
        # Epochs
        [5,10]]
    train_grid_linear = [
        # Dense layers
        [0, 10, 20, 64],
        # Epochs
        [5,10, 15, 20]]
    mini_train_grid_linear = [
        # Dense layers
        [64],
        # Epochs
        [5,10]]
    
    # train_points, train_labels, test_points, test_labels = LinearModel.get_data(PREPROCESSED_DATA_PATH)
    # grid = LinearTrainingGrid(*mini_train_grid_linear)
    # models = grid.train(train_points, train_labels)
    # best_model = grid.compare(models, test_points, test_labels)



    train_images,train_labels,test_images,test_labels = ConvModel.get_data(DATA_PATH)
    processed_train_images, processed_train_labels, processed_test_images, processed_test_labels = ConvModel.get_data(PREPROCESSED_DATA_PATH)
    tf.random.set_seed(1234)
    tf.config.experimental.enable_op_determinism()
    grid = ConvTrainingGrid(*mini_train_grid_conv)
    basic_models = grid.train(train_images, train_labels)
    processed_models = grid.train(processed_train_images, processed_train_labels)
    best_model = grid.compare(basic_models, test_images, test_labels)
    processed_best_model = grid.compare(processed_models, processed_test_images, processed_test_labels)

    
    
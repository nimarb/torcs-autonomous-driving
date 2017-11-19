import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

import os, os.path
current_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(current_dir, "..", "catkin_ws", "src", "img_to_sensor_data", "data")

class ImgToSensorCNN:
    """ ConvNet to infer distance and angle of a vehicle to the road centrefrom img data """

    def __init__(self):
        self.init_vars()

    def init_vars(self):
        self.img_width = 80
        self.img_height = 60
        self.num_train_set = 5400
        self.num_test_set = 100
        self.train_imgs = np.empty(self.num_train_set, dtype=object)
        self.test_imgs = np.empty(self.num_test_set, dtype=object)
        self.train_data = np.empty(self.num_train_set)
        self.test_data = np.empty(self.num_test_set)
        self.model = object
        self.batch_size = 32
        self.num_epochs = 3
        self.model_name = "model.hd5"

    def load_imgs(self):
        img_iter = 0
        for filename in glob.glob(data_dir + "/images/*.jpg"):
            img = cv2.imread(filename)
            if img_iter < self.num_train_set:
                #print("train imgs: " + str(img_iter))
                self.train_imgs[img_iter] = img
                img_iter += 1
            elif img_iter >= self.num_train_set:
                #print("test imgs: " + str(img_iter))
                self.test_imgs[img_iter - self.num_train_set] = img
                img_iter += 1
        print("All imgs loaded into np array")

    def load_labels(self):
        distance_array = np.load(data_dir + "/sensor/distance.npy")
        angle_array = np.load(data_dir + "/sensor/angle.npy")
        self.train_distance_array = distance_array[:self.num_train_set]
        self.test_distance_array = distance_array[self.num_train_set:self.num_train_set+self.num_test_set]
        self.train_angle_array = angle_array[:self.num_train_set]
        self.test_angle_array = angle_array[self.num_train_set:self.num_train_set+self.num_test_set]
        print("Loaded label arrays into np array")

    def check_equal_occurances(self, array):
        print("array contains " + str(array.size) + " elements in total and " + str(np.unique(array).size) + " unique entries")

    def plot_array_values(self, array):
        x = np.arange(array.size)
        plt.plot(x, array, linewidth=0.5)
        binwidth = 0.1
        #plt.hist(array, bins=np.arange(min(array), max(array) + binwidth, binwidth), linewidth=0.5)
        plt.show()

    def sort_data_into_classes(self):
        """actually not needed, working on a regression problem"""
        class_round_precision = 3
        num_classes_angles = np.around(self.train_angle_array, decimals=class_round_precision)
        num_classes_distances = np.around(self.train_distance_array, decimals=class_round_precision)

    def save_model(self):
        self.model.save(self.model_name)
        json = self.model.to_json()
        with open(self.model_name + ".json", 'w') as f:
            f.write(json)
        print("Saved model")

    def load_model(self):
        self.model = keras.models.load_model(self.model_name)
        print("Loaded model")

    def cnn_model(self):
        # loss= mean_squared_error, metrics=mean_absolute_error
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), padding="same",
                    input_shape=(self.img_height, self.img_width, 3), 
                    activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(64, kernel_size=(3, 3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Flatten())
        #self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(2, activation='relu'))

        data = np.empty((self.num_train_set, self.img_height, self.img_width, 3), dtype=object)
        for i in range (self.num_train_set):
            data[i, :, :, :] = self.train_imgs[i]

        train_target_vals = np.empty((self.num_train_set, 2))
        train_target_vals[:, 0] = self.train_angle_array
        train_target_vals[:, 1] = self.train_distance_array

        self.model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])
        self.model.fit(x=data, y=train_target_vals, 
                        batch_size=self.batch_size, epochs=self.num_epochs)
                        #validation_data=[self.test_angle_array, self.test_distance_array])

    def test_model(self):
        """ evaluate the loaded / trained model """
        data = np.empty((self.num_test_set, self.img_height, self.img_width, 3), dtype=object)
        for i in range (self.num_test_set):
            data[i, :, :, :] = self.test_imgs[i]

        train_target_vals = np.empty((self.num_test_set, 2))
        train_target_vals[:, 0] = self.test_angle_array
        train_target_vals[:, 1] = self.test_distance_array

        score = self.model.evaluate(x=data, y=train_target_vals, batch_size=self.batch_size)
        print(score)

    def preditct_test_pics(self):
        """ use the loaded/trained model to predict values for image data from the test set """
        data = np.empty((self.num_test_set, self.img_height, self.img_width, 3), dtype=object)
        for i in range (self.num_test_set):
            data[i, :, :, :] = self.test_imgs[i]

        prediction = self.model.predict(x=data)
        i = 0

        for val in prediction:
            print("angle val: " + str(self.test_angle_array[i]), "; pred: " + str(val[0]))
            print("dist val: " + str(self.test_distance_array[i]), "; pred: " + str(val[1]))
            i += 1
        

if __name__ == "__main__":
    cnn = ImgToSensorCNN()
    cnn.load_imgs()
    cnn.load_labels()
    cnn.cnn_model()
    cnn.save_model()
    #cnn.load_model()
    cnn.test_model()
    cnn.preditct_test_pics()

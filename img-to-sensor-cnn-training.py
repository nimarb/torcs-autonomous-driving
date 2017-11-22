import os, os.path
import glob
import json
import platform
from random import sample

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Activation, Dropout, LeakyReLU
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping, Callback

import numpy as np
import cv2
#import matplotlib.pyplot as plt

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

if "DigitsBoxBMW2" == platform.node():
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    DATA_DIR = os.path.join("/", "raid", "student_data", "PP_TORCS_LearnDrive1", "data")
else: 
    DATA_DIR = os.path.join(CURRENT_DIR, "..", "catkin_ws", "src", "img_to_sensor_data", "data")

class ImgToSensorCNN:
    """ ConvNet to infer distance and angle of a vehicle to the road centrefrom img data """

    def __init__(self):
        self.img_width = 80
        self.img_height = 60
        self.num_train_set = 5400
        self.num_test_set = 100
        self.train_imgs = np.empty(self.num_train_set, dtype=object)
        self.test_imgs = np.empty(self.num_test_set, dtype=object)
        self.model = object
        self.batch_size = 32
        self.num_epochs = 15
        self.model_name = "model.hd5"

    def set_test_set_in_percent(self, test_percent):
        """ Ability to set the test/training data size in percent of available img files """
        img_dir = os.path.join(DATA_DIR, "images")
        num_img = len([f for f in os.listdir(img_dir) if f.endswith(self.img_data_type) and os.path.isfile(os.path.join(img_dir, f))])
        self.num_test_set = num_img * (test_percent * 0.01)
        self.num_train_set = num_img - self.num_test_set

    def load_imgs(self):
        """ load all images into array, sorted by last modified time """
        tests = sample(range(0, (self.num_train_set + self.num_test_set) - 1), self.num_test_set)
        tests.sort()
        test_index = 0
        train_index = 0
        img_iter = 0
        for filename in sorted(glob.glob(DATA_DIR + "/images/*" + self.img_data_type), key=os.path.getmtime):
            img = cv2.imread(filename)
            #if img_iter < self.num_train_set:
            if tests[test_index] == img_iter:
                self.test_imgs[test_index] = img
                if self.num_test_set < test_index:
                    test_index += 1
                img_iter += 1
            #elif img_iter >= self.num_train_set:
            else:
                self.train_imgs[train_index] = img
                if self.num_train_set < train_index:
                    train_index += 1
                img_iter += 1
        print("All imgs loaded into np array")

    def load_labels(self):
        distance_array = np.load(DATA_DIR + "/sensor/distance.npy")
        angle_array = np.load(DATA_DIR + "/sensor/angle.npy")
        self.train_distance_array = distance_array[:self.num_train_set]
        self.test_distance_array = distance_array[self.num_train_set:self.num_train_set+self.num_test_set]
        self.train_angle_array = angle_array[:self.num_train_set]
        self.test_angle_array = angle_array[self.num_train_set:self.num_train_set+self.num_test_set]
        print("Loaded label arrays into np array")

    def test_shuffle_methods(self):
        """ tests array shuffling methods against each other """
        arr1 = np.arange(20, 30)
        arr2 = np.arange(50, 60)
        print(arr1)
        print(arr2)
        perm = np.random.permutation(10)
        print(arr1[perm])
        print(arr2[perm])

        arr1 = np.arange(20, 30)
        arr2 = np.arange(50, 60)
        rng_s = np.random.get_state()
        np.random.shuffle(arr1)
        np.random.set_state(rng_s)
        np.random.shuffle(arr2)
        print(arr1)
        print(arr2)

    def shuffle_data_arrays(self):
        rng_s = np.random.get_state()
        np.random.shuffle(self.train_angle_array)
        np.random.set_state(rng_s)
        np.random.shuffle(self.train_distance_array)
        np.random.set_state(rng_s)
        np.random.shuffle(self.train_imgs)
        np.random.set_state(rng_s)
        np.random.shuffle(self.test_angle_array)
        np.random.set_state(rng_s)
        np.random.shuffle(self.test_distance_array)
        np.random.set_state(rng_s)
        np.random.shuffle(self.test_imgs)
        print("Shuffled all data arrays!")

    def check_equal_occurances(self, array):
        print("array contains " + str(array.size) + " elements in total and " + str(np.unique(array).size) + " unique entries")

    def plot_array_values(self, array):
        x = np.arange(array.size)
        #plt.plot(x, array, linewidth=0.5)
        binwidth = 0.1
        #plt.hist(array, bins=np.arange(min(array), max(array) + binwidth, binwidth), linewidth=0.5)
        #plt.show()

    def sort_data_into_classes(self):
        """actually not needed, working on a regression problem"""
        class_round_precision = 3
        num_classes_angles = np.around(self.train_angle_array, decimals=class_round_precision)
        num_classes_distances = np.around(self.train_distance_array, decimals=class_round_precision)

    def save_model(self):
        self.model.save(self.model_name)
        json_str = self.model.to_json()
        json_str = json.dumps(json_str, indent=4, sort_keys=True)
        with open(self.model_name + ".json", 'w') as f:
            f.write(json_str)
        print("Saved model")

    def load_model(self):
        self.model = load_model(self.model_name)
        print("Loaded model")

    def cnn_model(self):
        # loss= mean_squared_error, metrics=mean_absolute_error
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), 
                    padding="same", 
                    input_shape=(self.img_height, self.img_width, 3)))
        self.model.add(LeakyReLU(alpha=0.1))
        #self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        #self.model.add(Conv2D(64, kernel_size=(3, 3)))
        #self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Flatten())
        #self.model.add(Dense(512))
        #self.model.add(Dense(2, activation="linear"))
        self.model.add(Dense(2))

        data = np.empty((self.num_train_set, self.img_height, self.img_width, 3), dtype=object)
        for i in range (self.num_train_set):
            data[i, :, :, :] = self.train_imgs[i]

        train_target_vals = np.empty((self.num_train_set, 2))
        train_target_vals[:, 0] = self.train_angle_array
        train_target_vals[:, 1] = self.train_distance_array

        cbs = []
        loss_hist = LossHistory()
        cbs.append(loss_hist)
        es = EarlyStopping(monitor='mean_absolute_error', min_delta=0.04)
        #cbs.append(es)

        self.model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=["mae"])
        self.model.fit(x=data, y=train_target_vals, 
                        batch_size=self.batch_size, epochs=self.num_epochs, 
                        callbacks=cbs)
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


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.metric = []
        self.epochs = 0

    def on_epoch_end(self, batch, logs={}):
        self.loss.append(logs.get("mean_squared_error"))
        self.metric.append(logs.get("mae"))
        self.epochs += 1
        

if __name__ == "__main__":
    train = True
    cnn = ImgToSensorCNN()
    cnn.load_imgs()
    cnn.load_labels()
    #cnn.test_shuffle_methods()
    cnn.shuffle_data_arrays()
    if True == train:
        cnn.cnn_model()
        cnn.save_model()
    else:
        cnn.load_model()
    cnn.test_model()
    cnn.preditct_test_pics()

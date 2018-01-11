import os
import os.path
import sys
import glob
import json
import platform
from random import sample
import random
import time

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Activation, Dropout, LeakyReLU, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping, Callback, CSVLogger, History
from keras.optimizers import Adam, Adamax

import numpy as np
import cv2

import tensorflow as tf

np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
tf.set_random_seed(42)
# import matplotlib.pyplot as plt

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_NAME = "data-olethros_road_1-2laps-640x480"
TEST_DATA_NAME = "data-cg_track_2-2laps-640x480"

DATA_NAMES = [
    "data-olethros_road_1-2laps-640x480",
    "data-aalborg-2laps-640x480",
    "data-alpine_1-2laps-640x480",
    "data-alpine_2-2laps-640x480",
    "data-brondehach-2laps-640x480",
    "data-cg_speedway_1-2laps-640x480",
    "data-cg_track_3-2laps-640x480",
    "data-corkscrew-2laps-640x480",
    "data-e_road-2laps-640x480",
    "data-etrack_1-2laps-640x480",
    "data-etrack_2-2laps-640x480",
    "data-etrack_3-2laps-640x480",
    "data-etrack_4-2laps-640x480",
    "data-etrack_6-2laps-640x480",
    "data-forza-2laps-640x480",
    "data-ruudskogen-2laps-640x480",
    #"data-spring-2laps-640x480",
    "data-street_1-2laps-640x480",
    "data-wheel_1-2laps-640x480"]
TEST_DATA_NAMES = [
    #"data-cg_track_3-2laps-640x480"]
    "data-cg_track_2-2laps-640x480",
    "data-wheel_2-2laps-640x480"]

if "data" in sys.argv[6]:
    print("A single track to train was given, it is: " + sys.argv[6])
    DATA_NAMES.clear()
    DATA_NAMES.append(sys.argv[6])

if "DigitsBoxBMW2" == platform.node():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    DATA_DIR = os.path.join(
        "/", "raid", "student_data", "PP_TORCS_LearnDrive1", DATA_NAME)
    TEST_DATA_DIR = os.path.join(
        "/", "raid", "student_data", "PP_TORCS_LearnDrive1", TEST_DATA_NAME)
    DATA_DIRS = []
    for track in DATA_NAMES:
        DATA_DIRS.append(
            os.path.join(
                "/", "raid", "student_data", "PP_TORCS_LearnDrive1", track))
    TEST_DATA_DIRS = []
    for track in TEST_DATA_NAMES:
        TEST_DATA_DIRS.append(
            os.path.join(
                "/", "raid", "student_data", "PP_TORCS_LearnDrive1", track))
else:
    DATA_DIR = os.path.join(
        CURRENT_DIR, "..", "collect_img_sensor_data", DATA_NAME)
    TEST_DATA_DIR = os.path.join(
        CURRENT_DIR, "..", "collect_img_sensor_data", TEST_DATA_NAME)
    DATA_DIRS = []
    for track in DATA_NAMES:
        DATA_DIRS.append(
            os.path.join(
                CURRENT_DIR, "..", "collect_img_sensor_data", track))
    TEST_DATA_DIRS = []
    for track in TEST_DATA_NAMES:
        TEST_DATA_DIRS.append(
            os.path.join(
                CURRENT_DIR, "..", "collect_img_sensor_data", track))


class ImgToSensorCNN:
    """Guess distance, angle of a vehicle using deep learning"""

    def __init__(
                self,
                model_name="learndrive-model",
                w=80,
                h=60,
                optimiser="adamax",
                model_architecture="simple",
                learning_rate=0.00001,
                dim_choice=1,
                camera_perspective="1st_no_hood",
                data_n=None):
        self.img_width = w
        self.img_height = h
        self.resize_imgs = True
        self.num_train_set = 1
        self.num_val_set = 1
        self.num_test_set = 20
        self.img_data_type = ".jpg"
        self.learning_rate = learning_rate
        self.distance_array = np.empty(0)
        self.angle_array = np.empty(0)
        self.train_imgs = np.empty(self.num_train_set, dtype=object)
        self.val_imgs = np.empty(self.num_val_set, dtype=object)
        #self.imgs = np.empty(
        #    self.num_train_set + self.num_val_set, dtype=object)
        self.imgs = np.empty(0, dtype=object)
        self.img_list = []
        self.model = object
        self.batch_size = 32
        self.num_epochs = 100
        self.loss_function = "mean_squared_error"
        self.metrics = "mae"
        self.model_name = model_name
        self.optimiser = optimiser
        self.model_architecture = model_architecture
        self.img_iter = 0
        # 0: only angle, 1: only distance, 2: angle&distance
        self.dim_choice = dim_choice
        self.camera_perspective = camera_perspective
        print("Data_n is: " + data_n)
        if data_n is not None:
            DATA_NAMES.clear()
            DATA_NAMES.append(data_n)

    def set_val_set_in_percent(self, val_percent):
        """Set the training/validation data size in percent of available img files
        
        Arguments
            val_percent: Integer, validation data size in percent of total data"""
        num_img = 0
        for track_dir in DATA_DIRS:
            img_dir = os.path.join(track_dir, "images")
            num_img += len([f for f in os.listdir(img_dir) if f.endswith(self.img_data_type) and os.path.isfile(os.path.join(img_dir, f))])
        self.num_val_set = round(num_img * (val_percent * 0.01))
        self.num_train_set = num_img - self.num_val_set
        self.train_imgs = np.empty(self.num_train_set, dtype=object)
        self.val_imgs = np.empty(self.num_val_set, dtype=object)

    def load_data(self):
        """Load img & sensor data and split into train/val/test set"""
        print("Loading " + str(self.num_val_set + self.num_train_set) + " imgs")
        for track_dir in DATA_DIRS:
            self.load_imgs(self.img_list, track_dir)
            self.load_labels(track_dir)
            print("img_list contains: " + str(len(self.img_list)) + " items")
            print("labels contain: " + str(self.distance_array.size) + " items")
        print("All imgs loaded into img_list")
        self.imgs = np.empty(len(self.img_list), dtype=object)
        for i in range(0, len(self.img_list)-1):
            self.imgs[i] = self.img_list[i]
        self.img_list = None
        print("All imgs loaded into np array")
        print("self.imgs np array contains: " + str(self.imgs.size) + " items")
        print("Labels contain: " + str(self.distance_array.size) + " items")
        self.split_into_train_val_set()

    def load_imgs(self, img_list, data_dir):
        """Load all images into list, sorted by file name (pad with zeros!)
        
        Arguments:
            img_list: 
            data_dir: """
        img_name_filter = glob.glob(
                            data_dir + "/images/*" + self.img_data_type)
        for filename in sorted(img_name_filter):
            img = cv2.imread(filename)
            if self.resize_imgs:
                (h, w, _) = img.shape
                if h != self.img_height and w != self.img_width:
                    factor = self.img_width / w
                    img = cv2.resize(img, None, fx=factor, fy=factor)
            img_list.append(img)
        print("All imgs of " + data_dir + " loaded into img_list")

    def load_imgs_with_rnd_split(self):
        """Load all imgs into array and randomly split into train/val
        DEPRECATED!!!"""
        tests = sample(
            range(0, (self.num_train_set + self.num_val_set) - 1),
            self.num_val_set)
        tests.sort()
        test_index = 0
        train_index = 0
        img_iter = 0
        img_name_filter = glob.glob(
                                DATA_DIR + "/images/*" + self.img_data_type)
        for filename in sorted(img_name_filter):
            img = cv2.imread(filename)
            if tests[test_index] == img_iter:
                self.val_imgs[test_index] = img
                if self.num_val_set < test_index:
                    test_index += 1
                img_iter += 1
            else:
                self.train_imgs[train_index] = img
                if self.num_train_set < train_index:
                    train_index += 1
                img_iter += 1
        print("All imgs randomly loaded into np array")

    def load_labels(self, data_dir):
        """Load recorded sensor data into numpy arrays"""
        _distance_array = np.load(data_dir + "/sensor/distance.npy")
        _angle_array = np.load(data_dir + "/sensor/angle.npy")
        self.distance_array = np.append(self.distance_array, _distance_array)
        self.angle_array = np.append(self.angle_array, _angle_array)
        print("Loaded label arrays of " + data_dir + " into np array")

    def load_test_set(self):
        """Load test set from a different track"""
        #_imgs = np.empty(self.num_test_set, dtype=object)
        _imgs_raw = []
        self.load_imgs(_imgs_raw, TEST_DATA_DIR)
        print("img_list contains: " + str(len(_imgs_raw)) + " items")
        _distance_array = np.load(TEST_DATA_DIR + "/sensor/distance.npy")
        _angle_array = np.load(TEST_DATA_DIR + "/sensor/angle.npy")
        _imgs = np.array(_imgs_raw)
        # Test data is from a different track but always a rnd subset
        self.shuffle_three_arrays_in_unison(
            _imgs, _angle_array, _distance_array)
        _imgs = _imgs[:self.num_test_set]
        self.test_imgs = np.empty(
            (self.num_test_set, self.img_height, self.img_width, 3),
            dtype=object)
        for i in range(self.num_test_set):
            self.test_imgs[i, :, :, :] = _imgs[i]

        self.test_vals = np.empty((self.num_test_set, 2))
        self.test_vals[:, 0] = _angle_array[:self.num_test_set]
        self.test_vals[:, 1] = _distance_array[:self.num_test_set]

    def split_into_train_val_set(self):
        """Splits loaded imgs&sensor data randomly in train/validation set"""
        self.shuffle_three_arrays_in_unison(
            self.imgs, self.distance_array, self.angle_array)

        total_set_size = self.num_val_set + self.num_train_set

        self.train_distance_array = self.distance_array[:self.num_train_set]
        self.val_distance_array = self.distance_array[
                                    self.num_train_set:total_set_size]
        self.train_angle_array = self.angle_array[:self.num_train_set]
        self.val_angle_array = self.angle_array[
                                    self.num_train_set:total_set_size]
        self.distance_array = None
        self.angle_array = None

        self.train_imgs = self.imgs[:self.num_train_set]
        self.val_imgs = self.imgs[self.num_train_set:total_set_size]
        self.imgs = None

    def test_shuffle_methods(self):
        """Tests array shuffling methods against each other"""
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

    def shuffle_three_arrays_in_unison(self, a, b, c):
        """Shuffles three given arrays in unison with each other"""
        rng_s = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_s)
        np.random.shuffle(b)
        np.random.set_state(rng_s)
        np.random.shuffle(c)

    def shuffle_data_arrays(self):
        """Shuffles all 3 data arrays (validation and train)"""
        self.shuffle_three_arrays_in_unison(
                    self.train_angle_array,
                    self.train_distance_array,
                    self.train_imgs)
        self.shuffle_three_arrays_in_unison(
                    self.val_angle_array,
                    self.val_distance_array,
                    self.val_imgs)
        print("Shuffled all data arrays!")

    def check_equal_occurances(self, array):
        """Prints the number of equal occurances in a np array
        
        Arguments:
            array: numpy array, array to check occurances in"""
        print(
            "array contains "
            + str(array.size)
            + " elements in total and "
            + str(np.unique(array).size)
            + " unique entries")

    def plot_array_values(self, array):
        """Plot the values of the given array

        Arguments
            array: numpy array, values to plot"""
        x = np.arange(array.size)
        # plt.plot(x, array, linewidth=0.5)
        binwidth = 0.1
        #plt.hist(
        #    array,
        #    bins=np.arange(min(array),
        #    max(array) + binwidth,
        #    binwidth),
        #    linewidth=0.5)
        # plt.show()

    def save(self):
        """Saves all data"""
        stamp = str(time.time()).split(".")[0]
        self.model_name = self.model_name + "-" + stamp[5:]
        self.save_model()
        self.save_metadata()

    def save_metadata(self):
        """Saves metadata for the current training in a json file"""
        metadata = {}
        metadata["img_width"] = self.img_width
        metadata["img_height"] = self.img_height
        metadata["img_data_type"] = self.img_data_type
        metadata["num_val_set"] = self.num_val_set
        metadata["num_test_set"] = self.num_test_set
        metadata["num_train_set"] = self.num_train_set
        metadata["num_epochs"] = self.num_epochs
        metadata["loss_function"] = self.loss_function
        metadata["metrics"] = self.metrics
        metadata["loss_hist"] = self.loss_hist.loss
        metadata["data_name"] = DATA_NAME
        metadata["test_data_name"] = TEST_DATA_NAME
        metadata["data_names"] = DATA_NAMES
        metadata["test_data_names"] = TEST_DATA_NAMES
        metadata["time_hist"] = self.time_hist.times
        metadata["train_loss_hist"] = self.fit_hist.history["loss"]
        metadata["train_mae_hist"] = self.fit_hist.history["mean_absolute_error"]
        metadata["val_loss_hist"] = self.fit_hist.history["val_loss"]
        metadata["val_mae_hist"] = self.fit_hist.history["val_mean_absolute_error"]
        metadata["test_loss"] = self.score[0]
        metadata["test_mae"] = self.score[1]
        metadata["optimiser"] = self.optimiser
        metadata["camera_perspective"] = self.camera_perspective
        metadata["model_architecture"] = self.model_architecture
        metadata["learning_rate"] = self.learning_rate
        if self.dim_choice == 2:
            metadata["dim_choice"] = "distance and angle"
        elif self.dim_choice == 1:
            metadata["dim_choice"] = "distance"
        elif self.dim_choice == 0:
            metadata["dim_choice"] = "angle"
        json_str = json.dumps(metadata)
        save_data_dir = os.path.join(
            "/", "raid", "student_data", "PP_TORCS_LearnDrive1", "models")
        with open(
                save_data_dir + self.model_name + "-metadata.json", "w") as f:
            f.write(json_str)
        print("Saved metadata")

    def save_model(self):
        """Saves the trained keras model to disk"""
        save_data_dir = os.path.join(
            "/", "raid", "student_data", "PP_TORCS_LearnDrive1", "models")
        self.model.save(save_data_dir + self.model_name + ".hd5")
        json_str = self.model.to_json()
        json_str = json.dumps(json_str, indent=4, sort_keys=True)
        with open(
                save_data_dir
                + self.model_name
                + "-architecture.json", 'w') as f:
            f.write(json_str)
        print("Saved model")

    def load_model(self, name=None):
        """Loads a previously saved keras model from disk

        Arguments:
            model: keras.model, filename"""
        if name:
            self.model_name = name
        self.model = load_model("../models/" + self.model_name + ".hd5")
        print("Loaded model")

    def cnn_model(self):
        """Creates a keras ConvNet model"""
        # loss= mean_squared_error, metrics=mean_absolute_error
        self.model = Sequential()
        if "alexnet" == self.model_architecture:
            self.cnn_alexnet()
        elif "alexnet_no_dropout" == self.model_architecture:
            self.cnn_alexnet_no_dropout()
        elif "tensorkart" == self.model_architecture:
            self.cnn_tensorkart()
        elif "simple" == self.model_architecture:
            self.cnn_simple()
        elif "simple_invcnv" == self.model_architecture:
            self.cnn_simple_invcnv()
        elif "simple_min1d" == self.model_architecture:
            self.cnn_simple_min1d()
        elif "simple_min2d" == self.model_architecture:
            self.cnn_simple_min2d()
        elif "simple_plu1d" == self.model_architecture:
            self.cnn_simple_plu1d()
        elif "simple_plu2d" == self.model_architecture:
            self.cnn_simple_plu2d()
        elif "simple_invcnv_adv" == self.model_architecture:
            self.cnn_simple_invcnv_adv()
        elif "simple_small" == self.model_architecture:
            self.cnn_simple_small()
        elif "simple_very_small" == self.model_architecture:
            self.cnn_simple_very_small()

        train_data = np.empty(
                (self.num_train_set, self.img_height, self.img_width, 3),
                dtype=object)
        for i in range(self.num_train_set):
            train_data[i, :, :, :] = self.train_imgs[i]

        val_data = np.empty(
                (self.num_val_set, self.img_height, self.img_width, 3),
                dtype=object)
        for i in range(self.num_val_set):
            val_data[i, :, :, :] = self.val_imgs[i]

        train_target_vals = np.empty((self.num_train_set, 2))
        train_target_vals[:, 0] = self.train_angle_array
        train_target_vals[:, 1] = self.train_distance_array

        val_target_vals = np.empty((self.num_val_set, 2))
        val_target_vals[:, 0] = self.val_angle_array
        val_target_vals[:, 1] = self.val_distance_array

        cbs = []
        self.loss_hist = LossHistory()
        self.fit_hist = History()
        self.time_hist = TimeHistory()
        cbs.append(self.fit_hist)
        cbs.append(self.loss_hist)
        cbs.append(self.time_hist)
        cbs.append(CSVLogger(self.model_name + ".csv", separator=','))
        es = EarlyStopping(monitor='mean_absolute_error', min_delta=0.04)
        # cbs.append(es)

        # other good optimiser: adam, rmsprop
        opti = Adamax(lr=self.learning_rate)
        self.model.compile(
            loss=self.loss_function,
            #optimizer=self.optimiser,
            optimizer=opti,
            metrics=[self.metrics])
        
        if self.dim_choice == 2:
        self.model.fit(
            x=train_data, y=train_target_vals, batch_size=self.batch_size,
            validation_data=(val_data, val_target_vals),
                epochs=self.num_epochs, callbacks=cbs)
        else:
            self.model.fit(
                x=train_data, y=train_target_vals[:, self.dim_choice], batch_size=self.batch_size,
                validation_data=(val_data, val_target_vals[:, self.dim_choice]),
            epochs=self.num_epochs, callbacks=cbs)

    def cnn_alexnet(self):
        """Uses the AlexNet network topology for the cnn model"""
        self.model.add(
            Conv2D(
                96, kernel_size=(11, 11), strides=(4, 4), padding="same",
                input_shape=(self.img_height, self.img_width, 3)))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(ZeroPadding2D(padding=(2, 2)))
        self.model.add(Conv2D(256, kernel_size=(5, 5), strides=(4, 4)))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(Conv2D(384, kernel_size=(3, 3)))
        self.model.add(Activation("relu"))
        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(Conv2D(384, kernel_size=(3, 3)))
        self.model.add(Activation("relu"))
        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(Conv2D(256, kernel_size=(3, 3)))
        self.model.add(Activation("relu"))
        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(4096))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))
        if 2 == self.dim_choice:
            self.model.add(Dense(2, activation='linear'))
        else:
            self.model.add(Dense(1, activation='linear'))

    def cnn_alexnet_no_dropout(self):
        """Uses the AlexNet network topology without dropout layers for the cnn model"""
        self.model.add(
            Conv2D(
                96, kernel_size=(11, 11), strides=(4, 4), padding="same",
                input_shape=(self.img_height, self.img_width, 3)))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(ZeroPadding2D(padding=(2, 2)))
        self.model.add(Conv2D(256, kernel_size=(5, 5), strides=(4, 4)))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(Conv2D(384, kernel_size=(3, 3)))
        self.model.add(Activation("relu"))
        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(Conv2D(384, kernel_size=(3, 3)))
        self.model.add(Activation("relu"))
        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(Conv2D(256, kernel_size=(3, 3)))
        self.model.add(Activation("relu"))
        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(4096))
        self.model.add(Activation("relu"))
        self.model.add(Dense(4096))
        self.model.add(Activation("relu"))
        if 2 == self.dim_choice:
            self.model.add(Dense(2, activation='linear'))
        else:
            self.model.add(Dense(1, activation='linear'))

    def cnn_tensorkart(self):
        """Uses NeuralKart/TensorKart model architecture"""
        # self.model.add(BatchNormalization(input_shape=(self.img_height, self.img_width, 3)))
        self.model.add(
            Conv2D(
                24,
                input_shape=(self.img_height, self.img_width, 3),
                                kernel_size=(5, 5),
                                strides=(2, 2),
                                activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(
            Conv2D(
                36,
                                kernel_size=(5, 5),
                                strides=(2, 2),
                                activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(
            Conv2D(
                48,
                                kernel_size=(5, 5),
                                strides=(2, 2),
                                activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(1164, activation='relu'))
        drop_out = 0.4
        self.model.add(Dropout(drop_out))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dropout(drop_out))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dropout(drop_out))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dropout(drop_out))
        if 2 == self.dim_choice:
            self.model.add(Dense(2, activation='linear'))
        else:
            self.model.add(Dense(1, activation='linear'))

    def cnn_simple_min1d(self):
        """Uses own simple model as cnn topology"""
        self.model.add(
            Conv2D(
                256,
            input_shape=(self.img_height, self.img_width, 3),
            kernel_size=(7, 7),
            padding='same',
            activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(
            Conv2D(
                256,
                                kernel_size=(5, 5),
                                activation='relu',
                                padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.mode.add(
            Conv2D(
                128,
                                kernel_size=(3, 3),
                                activation='relu',
                                padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        if 2 == self.dim_choice:
            self.model.add(Dense(2, activation='linear'))
        else:
            self.model.add(Dense(1, activation='linear'))

    def cnn_simple_min2d(self):
        """Uses own simple model as cnn topology"""
        self.model.add(Conv2D(256,
            input_shape=(self.img_height, self.img_width, 3),
            kernel_size=(7, 7),
            padding='same',
            activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256,
                                kernel_size=(5, 5),
                                activation='relu',
                                padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128,
                                kernel_size=(3, 3),
                                activation='relu',
                                padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        if 2 == self.dim_choice:
            self.model.add(Dense(2, activation='linear'))
        else:
            self.model.add(Dense(1, activation='linear'))

    def cnn_simple_plu1d(self):
        """Uses own simple model as cnn topology"""
        self.model.add(Conv2D(256,
            input_shape=(self.img_height, self.img_width, 3),
            kernel_size=(7, 7),
            padding='same',
            activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256,
                                kernel_size=(5, 5),
                                activation='relu',
                                padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128,
                                kernel_size=(3, 3),
                                activation='relu',
                                padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        if 2 == self.dim_choice:
            self.model.add(Dense(2, activation='linear'))
        else:
            self.model.add(Dense(1, activation='linear'))

    def cnn_simple_plu2d(self):
        """Uses own simple model as cnn topology"""
        self.model.add(Conv2D(256,
            input_shape=(self.img_height, self.img_width, 3),
            kernel_size=(7, 7),
            padding='same',
            activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256,
                                kernel_size=(5, 5),
                                activation='relu',
                                padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128,
                                kernel_size=(3, 3),
                                activation='relu',
                                padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        if 2 == self.dim_choice:
            self.model.add(Dense(2, activation='linear'))
        else:
            self.model.add(Dense(1, activation='linear'))

    def cnn_simple(self):
        """Uses own simple model as cnn topology"""
        self.model.add(Conv2D(256,
            input_shape=(self.img_height, self.img_width, 3),
            kernel_size=(7, 7),
            padding='same',
            activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256,
                                kernel_size=(5, 5),
                                activation='relu',
                                padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128,
                                kernel_size=(3, 3),
                                activation='relu',
                                padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        if 2 == self.dim_choice:
            self.model.add(Dense(2, activation='linear'))
        else:
            self.model.add(Dense(1, activation='linear'))

    def cnn_simple_invcnv(self):
        """Uses own simple model as cnn topology"""
        self.model.add(Conv2D(128,
            input_shape=(self.img_height, self.img_width, 3),
            kernel_size=(7, 7),
            padding='same',
            activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256,
                                kernel_size=(5, 5),
                                activation='relu',
                                padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(384,
                                kernel_size=(3, 3),
                                activation='relu',
                                padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        if 2 == self.dim_choice:
            self.model.add(Dense(2, activation='linear'))
        else:
            self.model.add(Dense(1, activation='linear'))

    def cnn_simple_invcnv_adv(self):
        """Uses own simple model as cnn topology"""
        self.model.add(Conv2D(128,
            input_shape=(self.img_height, self.img_width, 3),
            kernel_size=(7, 7),
            padding='same',
            activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256,
                                kernel_size=(5, 5),
                                activation='relu',
                                padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(384,
                                kernel_size=(3, 3),
                                activation='relu',
                                padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256,
                                kernel_size=(3, 3),
                                activation='relu',
                                padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        if 2 == self.dim_choice:
            self.model.add(Dense(2, activation='linear'))
        else:
            self.model.add(Dense(1, activation='linear'))

    def cnn_simple_small(self):
        self.model = Sequential()
        self.model.add(
            Conv2D(
                96,
                input_shape=(self.img_height, self.img_width, 3),
                kernel_size=(7, 7),
                padding='same',
                activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(7, 7), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        if 2 == self.dim_choice:
            self.model.add(Dense(2, activation='linear'))
        else:
            self.model.add(Dense(1, activation='linear'))

    def cnn_simple_very_small(self):
        self.model = Sequential()
        self.model.add(
            Conv2D(
                96,
                input_shape=(self.img_height, self.img_width, 3),
                kernel_size=(3, 3),
                padding='same',
                activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(7, 7), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        if 2 == self.dim_choice:
            self.model.add(Dense(2, activation='linear'))
        else:
            self.model.add(Dense(1, activation='linear'))

    def test_model(self):
        """Evaluate the loaded / trained model"""
        print("Evaluating model...")
        if self.dim_choice == 2:
        self.score = self.model.evaluate(
                                    x=self.test_imgs,
                                    # y=self.test_vals[:, 0],
                                    y=self.test_vals,
                                    batch_size=self.batch_size)
        else:
            self.score = self.model.evaluate(
                                        x=self.test_imgs,
                                        y=self.test_vals[:, self.dim_choice],
                                        batch_size=self.batch_size)

        print("Model evaluated, score is:")
        print(self.score)

    def preditct_test_pics(self):
        """Use model to predict values for image data from the test set"""
        data = np.empty(
                (1, self.img_height, self.img_width, 3),
                #(self.num_test_set, self.img_height, self.img_width, 3),
                dtype=object)
        #for i in range(self.num_test_set):
        for i in range(1):
            data[i, :, :, :] = self.test_imgs[i]
        #i = 0
        #data[i, :, :, :] = self.test_imgs[i]

        t1 = time.time()
        prediction = self.model.predict(x=data)
        dt = time.time() - t1

        pred_descr = ["angl", "dist"]
        i = 0
        for val in prediction:
            if self.dim_choice == 2:
            print(
                    "angl val: "
                + str(self.test_vals[i, 0])
                + "; pred: "
                + str(val[0]))
            print(
                "dist val: "
                + str(self.test_vals[i, 1])
                + "; pred: "
                + str(val[1]))
            else:
                print(
                    pred_descr[self.dim_choice]
                    + " val: "
                    + str(self.test_vals[i, self.dim_choice])
                    + "; pred: "
                    + str(val[0]))
            i += 1


class LossHistory(Callback):
    """Class saving the per epoch metrics generated by .fit"""

    def on_train_begin(self, logs={}):
        """Before first epoch, define variables"""
        self.loss = []
        self.metric = []
        self.epochs = 0

    def on_epoch_end(self, batch, logs={}):
        """Logs after each epoch; ATTENTION, log property has to be updated manually"""
        self.loss.append(logs.get("loss"))
        self.metric.append(logs.get("mae"))
        self.epochs += 1


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        

if __name__ == "__main__":
    if "train" == sys.argv[1]:
        train = True
    elif "test" == sys.argv[1]:
        train = False
    if len(sys.argv) == 2:
        if sys.argv[2]:
            cnn = ImgToSensorCNN(sys.argv[2])
    elif len(sys.argv) == 4:
        cnn = ImgToSensorCNN(w=int(sys.argv[2]), h=int(sys.argv[3]))
    elif len(sys.argv) == 5:
        cnn = ImgToSensorCNN(
            w=int(sys.argv[2]), h=int(sys.argv[3]), model_architecture=sys.argv[4])
    elif len(sys.argv) == 7:
        cnn = ImgToSensorCNN(
            w=int(sys.argv[2]), h=int(sys.argv[3]), model_architecture=sys.argv[4], camera_perspective=sys.argv[5], data_n=sys.argv[6])

    cnn.set_val_set_in_percent(10)
    if train:
        DATA_NAMES.clear()
        DATA_NAMES.append(sys.argv[6])
        cnn.load_data()
        cnn.shuffle_data_arrays()
        cnn.cnn_model()
        cnn.test_model()
        # cnn.preditct_test_pics()
        cnn.save()
    else:
        cnn.load_model("learndrive-model-89752")
        cnn.test_model()
        cnn.preditct_test_pics()

import os
import os.path
import sys
import glob
import json
import platform
# from random import sample
import random
import time

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import EarlyStopping, Callback, CSVLogger, History
from keras.optimizers import Adam, Adamax

import cnn_models

np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
tf.set_random_seed(42)

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_NAMES = [
    "data-olethros_road_1-2laps-640x480",
    #"data-aalborg-2laps-640x480",
    #"data-alpine_1-2laps-640x480",
    #"data-alpine_2-2laps-640x480",
    #"data-brondehach-2laps-640x480",
    #"data-cg_speedway_1-2laps-640x480",
    #"data-cg_track_3-2laps-640x480",
    #"data-corkscrew-2laps-640x480",
    #"data-e_road-2laps-640x480",
    #"data-etrack_1-2laps-640x480",
    #"data-etrack_2-2laps-640x480",
    #"data-etrack_3-2laps-640x480",
    #"data-etrack_4-2laps-640x480",
    #"data-etrack_6-2laps-640x480",
    #"data-forza-2laps-640x480",
    #"data-ruudskogen-2laps-640x480",
    ###"data-spring-2laps-640x480",
    #"data-street_1-2laps-640x480",
    "data-wheel_1-2laps-640x480"]
TEST_DATA_NAMES = [
    #"data-cg_track_3-2laps-640x480"]
    "data-cg_track_2-2laps-640x480",
    "data-wheel_2-2laps-640x480"]

if len(sys.argv) > 6:
    if "data" in sys.argv[6]:
        print("A single track to train was given, it is: " + sys.argv[6])
        DATA_NAMES.clear()
        DATA_NAMES.append(sys.argv[6])

if "DigitsBoxBMW2" == platform.node():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
                dim_choice=2,
                camera_perspective="1st_no_hood",
                data_n=None,
                normalise_imgs=True,
                normalise_arrays=False,
                top_region_cropped=False,
                img_depth=3,
                colourspace="rgb",
                hsv_layer="s",
                data_thinning_enabled=False,
                thinning_min_delta=0.005,
                thinning_avg_over_eles=5):
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
        self.imgs = np.empty(0, dtype=object)
        self.img_list = []
        self.model = object
        self.batch_size = 32
        self.num_epochs = 50
        self.loss_function = "mean_squared_error"
        self.metrics = "mae"
        self.model_name = model_name
        self.optimiser = optimiser
        self.model_architecture = model_architecture
        self.img_iter = 0
        # 0: only angle, 1: only distance, 2: angle&distance
        self.dim_choice = dim_choice
        self.camera_perspective = camera_perspective
        self.normalise_imgs = normalise_imgs
        self.normalise_arrays = normalise_arrays
        self.top_region_cropped = top_region_cropped
        self.top_crop_factor = 0.4
        if colourspace == "hsv":
            img_depth = 1
            self.hsv_layer = hsv_layer
        self.colourspace = colourspace
        self.img_depth = img_depth
        self.data_thinning_enabled = data_thinning_enabled
        self.thinning_min_delta = thinning_min_delta
        self.thinning_avg_over_eles = thinning_avg_over_eles
        if data_n is not None:
            print("Data_n is: " + data_n)
            DATA_NAMES.clear()
            DATA_NAMES.append(data_n)

    def set_val_set_in_percent(self, val_percent):
        """Set the training/validation data size in percent of available img files
        
        Arguments
            val_percent: int, validation data size in percent of total data"""
        num_img = 0
        for track_dir in DATA_DIRS:
            img_dir = os.path.join(track_dir, "images")
            num_img += len([f for f in os.listdir(img_dir) if f.endswith(self.img_data_type) and os.path.isfile(os.path.join(img_dir, f))])
        self.num_val_set = round(num_img * (val_percent * 0.01))
        self.num_train_set = num_img - self.num_val_set
        self.train_imgs = np.empty(self.num_train_set, dtype=object)
        self.val_imgs = np.empty(self.num_val_set, dtype=object)
        self.val_percent = val_percent

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

    def load_imgs(self, img_list, data_dir, a=0, b=1):
        """Load all images into list, sorted by file name (pad with zeros!)
        
        Arguments:
            img_list:
            data_dir:
            a: int, min value of img to normalise to
            b: int, max value of img to normalise to"""
        img_name_filter = glob.glob(
                            data_dir + "/images/*" + self.img_data_type)
        for filename in sorted(img_name_filter):
            img = cv2.imread(filename)
            if self.normalise_imgs:
                norm_img = np.zeros(img.shape)
                norm_img = cv2.normalize(
                    img,
                    norm_img,
                    alpha=a,
                    beta=b,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_32F)
                img = norm_img
            if self.resize_imgs:
                (h, w, _) = img.shape
                if h != self.img_height and w != self.img_width:
                    factor = self.img_width / w
                    img = cv2.resize(img, None, fx=factor, fy=factor)
            if self.top_region_cropped:
                top_crop = int(self.top_crop_factor * self.img_height)
                img = img[top_crop:self.img_height, 0:self.img_width]
                self.img_height = self.img_height - top_crop
            if self.colourspace == "hsv":
                if 1 == self.img_depth:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    img = img[:, :, 1:2]
            img_list.append(img)
        print("All imgs of " + data_dir + " loaded into img_list")

    def load_labels(self, data_dir, a=-1, b=1):
        """Load recorded sensor data into numpy arrays

        Arguments:
            data_dir: string, data dir containing the sensor folder
            a: int, minimal value to normalise to
            b: int, maximal value to normalise to """
        _distance_array = np.load(data_dir + "/sensor/distance.npy")
        _angle_array = np.load(data_dir + "/sensor/angle.npy")
        # TODO:
        # if self.normalise_arrays:
            
        self.distance_array = np.append(self.distance_array, _distance_array)
        self.angle_array = np.append(self.angle_array, _angle_array)
        print("Loaded label arrays of " + data_dir + " into np array")

    def load_test_set(self):
        """Load test set from a different track"""
        _imgs_raw = []
        _distance_array = np.empty(0)
        _angle_array = np.empty(0)
        for track in TEST_DATA_DIRS:
            self.load_imgs(_imgs_raw, track)
            _distance_array = np.append(
                _distance_array, np.load(track + "/sensor/distance.npy"))
            _angle_array = np.append(
                _angle_array, np.load(track + "/sensor/angle.npy"))

        print("img_list contains: " + str(len(_imgs_raw)) + " items")
        print("test labels contain: " + str(_distance_array.size) + " items")
        if len(_imgs_raw) == _distance_array.size:
            self.num_test_set = len(_imgs_raw)
        else:
            print("ERROR: npy array size and num_test_imgs doesnt match")

        self.test_imgs = np.empty(
            (
                self.num_test_set,
                self.img_height,
                self.img_width,
                self.img_depth),
            dtype=object)

        for i in range(0, self.num_test_set):
            self.test_imgs[i, :, :, :] = _imgs_raw[i]

        _imgs_raw = None
        # _imgs = np.empty(self.num_test_set, dtype=object)
        # _imgs = np.array(_imgs_raw)
        # print("_imgs array contains: " + str(_imgs.size) + " items")
        # _imgs = _imgs[:self.num_test_set]
        # for i in range(self.num_test_set):
        #    self.test_imgs[i, :, :, :] = _imgs[i]
            
        # Test data is from a different track but always a rnd subset
        self.shuffle_three_arrays_in_unison(
            self.test_imgs, _angle_array, _distance_array)

        self.test_vals = np.empty((self.num_test_set, 2))
        self.test_vals[:, 0] = _angle_array[:self.num_test_set]
        self.test_vals[:, 1] = _distance_array[:self.num_test_set]

    def thin_data_set(
            self, distance_array, angle_array, img_array,
            min_delta=0.005, avg_nr_eles=5):
        """
        
        Arguments:
            distance_array: numpy array,
            angle_array: numpy array,
            img_array: numpy array,
            min_delta: float,
            avg_nr_eles: int,"""

        # if value in dist_array is close to zero (min_delta dist)
        # and no change in value greater than min_delta over
        # avg_nr_else number of frames, then delete avg_nr_frames-1
        # from data set at i+1
        entries_to_delete = []
        for i in range(0, distance_array.size-avg_nr_eles):
            if np.absolute(distance_array[i]) > min_delta:
                dist_tmp_mean = np.mean(distance_array[i:i+avg_nr_eles])
                diffs = np.subtract(
                    dist_tmp_mean, distance_array[i:i+avg_nr_eles])
                diffs = np.absolute(diffs)
                diff_detected_counter = 0
                for j in range(diffs.size):
                    if diffs[j] > min_delta:
                        diff_detected_counter += 1
                if diff_detected_counter >= avg_nr_eles:
                    for j in range(1, avg_nr_eles):
                        entries_to_delete.append(i+j)

        entries_to_delete_arr = np.empty(len(entries_to_delete))
        for i in range(len(entries_to_delete)):
            entries_to_delete_arr[i] = entries_to_delete[i]
        distance_array = np.delete(distance_array, entries_to_delete_arr)
        angle_array = np.delete(angle_array, entries_to_delete_arr)
        img_array = np.delete(img_array, entries_to_delete_arr)
        print(
            "Data set thinning deleted a total of: "
            + str(entries_to_delete_arr.size)
            + " entries")
        print(
            "New array sizes\n"
            + "dist: "
            + str(self.distance_array.size)
            + "\ndist2: "
            + str(distance_array.size))

    def split_into_train_val_set(self):
        """Splits loaded imgs&sensor data randomly in train/validation set"""
        self.shuffle_three_arrays_in_unison(
            self.imgs, self.distance_array, self.angle_array)

        if self.data_thinning_enabled:
            self.thin_data_set(
                self.distance_array, self.angle_array, self.imgs)
            num_img = self.imgs.size
            self.num_val_set = round(num_img * (self.val_percent * 0.01))
            self.num_train_set = num_img - self.num_val_set
            self.train_imgs = np.empty(self.num_train_set, dtype=object)
            self.val_imgs = np.empty(self.num_val_set, dtype=object)
            print("Resized arrays to match thinned data set.")

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
        """Shuffles three given arrays in unison with each other
        
        Arguments:
            a: numpy array
            b: numpy array
            c: numpy array"""
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
        plt.plot(x, array, linewidth=0.5)
        binwidth = 0.1
        plt.hist(
            array,
            bins=np.arange(
                min(array), max(array) + binwidth, binwidth),
            linewidth=0.5)
        plt.show()
    
    def print_array_stats(self, distance_array, angle_array):
        """Print stats of input distance and angle arrays

        Arguments:
            distance_array: numpy array
            angle_array: numpy array"""
        print(
            "Distance array: max="
            + str(max(distance_array))
            + "; min="
            + str(min(distance_array))
            + "; avg: "
            + str(np.mean(distance_array)))
        print(
            "Angle array: max="
            + str(max(angle_array))
            + "; min="
            + str(min(angle_array))
            + "; avg: "
            + str(np.mean(angle_array)))

    def visualise_data_connection(
            self, img_array, distance_array=None, angle_array=None):
        """Shows images and corresponding distance and angle values

        Arguments:
            distance_array: numpy array, contains the distance values
            angle_array: numpy array, contains the angle values"""

        print("Showing images with assigned values side by side...")
        if angle_array is None:
            angle_array = np.zeros(shape=distance_array.shape)
            print("No angle array given, angle values will be zero")
        if distance_array is None:
            distance_array = np.zeros(shape=angle_array.shape)
            print("No distance array given, distance values will be zero")
        self.print_array_stats(distance_array, angle_array)
        i = 0
        for img in img_array:
            print(
                "Dist: "
                + str(distance_array[i])
                + ";\tangl: "
                + str(angle_array[i]))
            print("img shape: " + str(img.shape))
            cv2.imshow("Img" + str(i), img_array[i])
            cv2.waitKey()
            cv2.destroyWindow("Img" + str(i))
            i += 1

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
        metadata["normalise_imgs"] = self.normalise_imgs
        metadata["normalise_arrays"] = self.normalise_arrays
        metadata["top_region_cropped"] = self.top_region_cropped
        metadata["top_crop_factor"] = self.top_crop_factor
        metadata["img_depth"] = self.img_depth
        metadata["colourspace"] = self.colourspace
        metadata["hsv_layer"] = self.hsv_layer
        metadata["data_thinning_enabled"] = self.data_thinning_enabled
        metadata["data_thinning_min_delta"] = self.data_thinning_min_delta
        metadata["data_thinning_avg_over_eles"] = self.data_thinning_avg_over_eles
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
        self.model.save(save_data_dir + "/" + self.model_name + ".hd5")
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
        
        _model_path = os.path.join(
            "/home/nb/progs/torcs-autonomous-driving/src/models",
            self.model_name + ".hd5")
        self.model = load_model(_model_path)
        print("Loaded model")

    def load_metadata(self, model_name=None):
        """ """

        if model_name:
            self.model_name = model_name

        if "DigitsBoxBMW2" == platform.node():
            model_dir = os.path.join(
                "/", "raid", "student_data", "PP_TORCS_LearnDrive1", "models")
            _model_metadata_path = os.path.join(
                model_dir, self.model_name + "-metadata.json")
        else:
            _model_metadata_path = os.path.join(
                "/home/nb/progs/torcs-autonomous-driving/src/models",
                self.model_name + "-metadata.json")

        metadata = json.load(open(_model_metadata_path))
        try:
            self.img_width = metadata["img_width"]
            self.img_height = metadata["img_height"]
            self.img_data_type = metadata["img_data_type"]
            self.camera_perspective = metadata["camera_perspective"]
            self.model_architecture = metadata["model_architecture"]
            self.normalise_imgs = metadata["normalise_imgs"]
            self.normalise_arrays = metadata["normalise_arrays"]
            self.top_regio_cropped = metadata["top_region_cropped"]
            self.top_crop_factor = metadata["top_crop_factor"]
            self.img_depth = metadata["img_depth"]
            self.colourspace = metadata["colourspace"]
            self.hsv_layer = metadata["hsv_layer"]
            if metadata["dim_choice"] == "distance and angle":
                self.dim_choice = 2
            elif metadata["dim_choice"] == "distance":
                self.dim_choice = 1
            elif metadata["dim_choice"] == "angle":
                self.dim_choice = 0
        except KeyError:
            print("several keys not found but start anyway")
            pass

    def cnn_model(self):
        """Creates a keras ConvNet model"""
        # loss= mean_squared_error, metrics=mean_absolute_error
        if "alexnet" == self.model_architecture:
            self.model = cnn_models.alexnet(
                self.img_height,
                self.img_width,
                self.img_depth,
                self.dim_choice)
        elif "alexnet_no_dropout" == self.model_architecture:
            self.model = cnn_models.alexnet_no_dropout(
                self.img_height,
                self.img_width,
                self.img_depth,
                self.dim_choice)
            self.cnn_alexnet_no_dropout()
        elif "tensorkart" == self.model_architecture:
            self.model = cnn_models.tensorkart(
                self.img_height,
                self.img_width,
                self.img_depth,
                self.dim_choice)
        elif "simple" == self.model_architecture:
            self.model = cnn_models.simple(
                self.img_height,
                self.img_width,
                self.img_depth,
                self.dim_choice)
        elif "simple_invcnv" == self.model_architecture:
            self.model = cnn_models.simple_invcnv(
                self.img_height,
                self.img_width,
                self.img_depth,
                self.dim_choice)
        elif "simple_min1d" == self.model_architecture:
            self.model = cnn_models.simple_min1d(
                self.img_height,
                self.img_width,
                self.img_depth,
                self.dim_choice)
        elif "simple_min2d" == self.model_architecture:
            self.model = cnn_models.simple_min2d(
                self.img_height,
                self.img_width,
                self.img_depth,
                self.dim_choice)
        elif "simple_plu1d" == self.model_architecture:
            self.model = cnn_models.simple_plu1d(
                self.img_height,
                self.img_width,
                self.img_depth,
                self.dim_choice)
        elif "simple_plu2d" == self.model_architecture:
            self.model = cnn_models.simple_plu2d(
                self.img_height,
                self.img_width,
                self.img_depth,
                self.dim_choice)
        elif "simple_invcnv_adv" == self.model_architecture:
            self.model = cnn_models.simple_invcnv_adv(
                self.img_height,
                self.img_width,
                self.img_depth,
                self.dim_choice)
        elif "simple_small" == self.model_architecture:
            self.model = cnn_models.simple_small(
                self.img_height,
                self.img_width,
                self.img_depth,
                self.dim_choice)
        elif "simple_very_small" == self.model_architecture:
            self.model = cnn_models.simple_very_small(
                self.img_height,
                self.img_width,
                self.img_depth,
                self.dim_choice)
        elif "simple_small_leaky_relu" == self.model_architecture:
            self.model = cnn_models.simple_small_leaky_relu(
                self.img_height,
                self.img_width,
                self.img_depth,
                self.dim_choice)
        elif "simple_very_small_3l" == self.model_architecture:
            self.model = cnn_models.simple_very_small_3l(
                self.img_height,
                self.img_width,
                self.img_depth,
                self.dim_choice)

        train_data = np.empty(
                (
                    self.num_train_set,
                    self.img_height,
                    self.img_width,
                    self.img_depth),
                dtype=object)
        for i in range(self.num_train_set):
            train_data[i, :, :, :] = self.train_imgs[i]

        val_data = np.empty(
                (
                    self.num_val_set,
                    self.img_height,
                    self.img_width,
                    self.img_depth),
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
            # optimizer=self.optimiser,
            optimizer=opti,
            metrics=[self.metrics])
        
        if self.dim_choice == 2:
            self.model.fit(
                x=train_data, y=train_target_vals, batch_size=self.batch_size,
                validation_data=(val_data, val_target_vals),
                epochs=self.num_epochs, callbacks=cbs)
        else:
            self.model.fit(
                x=train_data,
                y=train_target_vals[:, self.dim_choice],
                batch_size=self.batch_size,
                validation_data=(
                    val_data, val_target_vals[:, self.dim_choice]),
                epochs=self.num_epochs,
                callbacks=cbs)

    def test_model(self):
        """Evaluate the loaded / trained model"""
        print("Evaluating model: " + self.model_name)
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

        print("Model evaluated, the score is: ")
        print(self.score)

    def preditct_test_pics(self, num_imgs_to_predict=0):
        """Use model to predict values for image data from the test set
        
        Arguments:
            num_imgs_to_predict: int, number of imgs to run the prediction on,
                                    if zero -> run prediction on all imgs"""
        print("Predicting images from: " + self.model_name)

        if 0 < num_imgs_to_predict:
            data = np.empty(
                    (
                        num_imgs_to_predict,
                        self.img_height,
                        self.img_width,
                        self.img_depth),
                    dtype=object)
            for j in range(num_imgs_to_predict):
                data[j, :, :, :] = self.test_imgs[j, :, :, :]
            self.num_test_set = num_imgs_to_predict

        print(
            "Predicting test pictures, "
            + str(self.num_test_set)
            + " pics to predict...")

        t1 = time.time()
        if 0 == num_imgs_to_predict:
            prediction = self.model.predict(x=self.test_imgs, verbose=1)
        else:
            prediction = self.model.predict(x=data, verbose=1)

        dt = time.time() - t1
        print("Time needed for prediction: " + str(dt))
        print(
            "Avg. time needed for prediction per img: "
            + str(dt/self.num_test_set))

        pred_descr = ["angl", "dist"]
        i = 0
        for val in prediction:
            if self.dim_choice == 2:
                print(
                    "angl val: "
                    + str(self.test_vals[i, 0])
                    + ";\tpred: "
                    + str(val[0]))
                print(
                    "dist val: "
                    + str(self.test_vals[i, 1])
                    + ";\tpred:"
                    + str(val[1]))
            else:
                print(
                    pred_descr[self.dim_choice]
                    + " val: "
                    + str(self.test_vals[i, self.dim_choice])
                    + ";\tpred: "
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
    elif "-h" == sys.argv[1]:
        print(
            "Usage: python train-dnn.py INTEND PARAMETRES"
            + "\tINTEND: \n\t\ttest: to test a model"
            + "\n\t\ttrain: to train a model")
    else:
        print("ERROR: provide train or test as first argument, -h for help")
    if train:
        if len(sys.argv) == 4:
            cnn = ImgToSensorCNN(w=int(sys.argv[2]), h=int(sys.argv[3]))
        elif len(sys.argv) == 5:
            cnn = ImgToSensorCNN(
                w=int(sys.argv[2]),
                h=int(sys.argv[3]),
                model_architecture=sys.argv[4])
        elif len(sys.argv) == 7:
            cnn = ImgToSensorCNN(
                w=int(sys.argv[2]),
                h=int(sys.argv[3]),
                model_architecture=sys.argv[4],
                camera_perspective=sys.argv[5],
                data_n=sys.argv[6])
    else:
        cnn = ImgToSensorCNN()

    cnn.set_val_set_in_percent(10)
    if train:
        cnn.load_data()
        cnn.split_into_train_val_set()
        cnn.shuffle_data_arrays()
        cnn.print_array_stats(
            cnn.train_distance_array, cnn.train_angle_array)
        #cnn.visualise_data_connection(
        #    cnn.train_imgs,
        #    cnn.distance_array,
        #    cnn.angle_array)
        cnn.cnn_model()
        cnn.load_test_set()
        cnn.test_model()
        cnn.preditct_test_pics()
        cnn.save()
    else:
        MODEL_DIR = "model_arch-val_mae-comp-large/"
        cnn.load_model(MODEL_DIR + "modelslearndrive-model-21063")
        cnn.load_metadata()
        cnn.load_test_set()
        #if cnn.dim_choice == 2:
            #cnn.visualise_data_connection(
            #    img_array=cnn.test_imgs,
            #    distance_array=cnn.test_vals[:, 0],
            #    angle_array=cnn.test_vals[:, 1])
        #else:
            #cnn.visualise_data_connection(
            #    img_array=cnn.test_imgs,
            #    distance_array=cnn.test_vals[:, cnn.dim_choice])
        cnn.test_model()
        cnn.preditct_test_pics(10)

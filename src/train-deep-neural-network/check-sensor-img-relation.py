import numpy as np
import os

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

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
    "data-wheel_1-2laps-640x480",
    "data-cg_track_2-2laps-640x480",
    "data-wheel_2-2laps-640x480"]

DATA_DIRS = []
for track in DATA_NAMES:
    DATA_DIRS.append(
        os.path.join(
            CURRENT_DIR, "..", "collect_img_sensor_data", track))

for cur_dir in DATA_DIRS:
    angle = np.load(os.path.join(cur_dir, "sensor", "angle.npy"))
    distance = np.load(os.path.join(cur_dir, "sensor", "distance.npy"))
    img_dir = os.path.join(cur_dir, "images")
    num_img = len([f for f in os.listdir(img_dir) if f.endswith(".jpg") and os.path.isfile(os.path.join(img_dir, f))])

    print("Track: " + cur_dir)
    print("Angle size: " + str(angle.size))
    print("Distance size: " + str(distance.size))
    print("Img count: " + str(num_img))

    if angle.size != num_img:
        print("ERROR CHECK PLEASE")

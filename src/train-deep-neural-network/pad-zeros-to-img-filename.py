import os
path = '/home/nb/progs/torcs-autonomous-driving/src/collect_img_sensor_data/data-track1-120x160/images/'
for filename in os.listdir(path):
    num = filename[:-4]
    #prefix, num = filename[:-4].split('_')
    num = num.zfill(4)
    #new_filename = prefix + "_" + num + ".jpg"
    new_filename = num + ".jpg"
    os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
from keras.utils import plot_model
from keras.models import load_model
from keras.utils.vis_utils import model_to_dot

MODELDIR = "/home/nb/progs/torcs-autonomous-driving/src/models/"

#modelname = "model_arch-val_mae-comp-large/modelslearndive-model-52797.hd5"
modelname = "model-52797"
modelpath = "/home/nb/progs/torcs-autonomous-driving/src/models/model_arch-val_mae-comp-large/modelslearndrive-model-52797.hd5"

model = load_model(modelpath)

plot_model(model, to_file=modelname+".png")
model_to_dot(model)

# Architecture observations
 - leaky relu (alpha=0.3) is better than relu for first Conv2d layer; less loss, mae after 7 epochs (55 vs 35)
 - optimiser rmsprop vs adam: rmsprop has good MAE (metric) but very bad MSE (loss)

# Keras
 - minibatches are shuffled by default

# Interesing links / sources
 - https://de.mathworks.com/help/nnet/examples/train-a-convolutional-neural-network-for-regression.html

# Requirements
 * [X] randomly select train/test data from total data set
 * [ ] ability to only use every xth data sample (i.e. only every third frame)
 * [X] ability set the train/test split in percent
 * [X] record time needed per epoch
 * [ ] ability to normalise the input images to values between [0....1] before input
 * [ ] clearly define and split data set into **train**, **validation (dev)**, **test** -> Andrew Ng talk relevant!

 # Meeting 2017-11-23
 ## Nengo
 ## Img-to-sensor
  - Most important variables?
  - How many imgs
 ## ROS
  - Script different maps?
  - Which driver to model?
  - pytry nengo torcs uni waterloo

# Things to vary
 * colour/bw image
 * image sizes
 * batch sizes? 
 * amount of training data
 * mix training data from multiple tracks
 * train only for one value: angle, displacement and determine which is easier to train for
 * relu instead of leakyrelu
 * activation func for every conv2d
 * **increase learning rate** -> how?
 * **change optimiser** -> maybe caught up in a local minima? SGD
 * change loss function: MSE? MAE?
 * the other group considers to use a pre trained network (VGG19 on ImgNet data), I think it makes not much sense but if it is easy to try, maybe do so...

# Notes
 * kleine Zielwerte -> langsames lernen
 * 













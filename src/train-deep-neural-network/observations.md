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
 * [ ] **record time needed per epoch**


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
 * 












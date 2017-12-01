# Requirements

 * [X] collect img, angle, distance data
 * [ ] automatically collect data from multiple tracks
 * [X] collect a predefined number of samples
 * [ ] after predefined num, continue to collect samples and write to new file
 * [X] mode which collects samples until end of track (or until ctrl+c)

# File type
 - Currently saving as/working with jpg files -> loss -> artefacts present
   - Will using png images (not lossy) improve accuracy of model?

# Data capture settings
 - ca. 600pictures/min -> ~10fps

# Further Ideas
 * collect training data from TORCS drivers -> a lot faster apparently
 * collect data with different vehicle views: hood, 3rd person etc: possible simultaneously?
 * make number of laps selectable during training
 
The Task_1 folder contains files/scripts relevant for Task 1 of the challenge and the Task_2 folder contains files/scripts for Task 2. For both Tasks, the directory variable "root_dir" in the notebook will require changing to reflect a user's personal directory path structure.

For Task_1, a jupyter notebook is provided that illustrates the model/data pre-processing steps taken to train a modified ResNet to distinguish between
good and bad tracks. The model weights are also saved as a compressed file for ease of inference.

For Task_2, due to the high volume of data and computation required for this approach to task 2, you will likely not be able to run it on a personal computer (it will take too long) unless you have a personal GPU. Detailed in the python notebook, this approach utilizes an unsupervised learning approach to detect patterns in the data - the code checks univariate and multivariate criteria to compile similarity scores for each sortie correlated to each labeled maneuver example. The algorithm predicts the maneuver(s) based on how high the similarity scores are.

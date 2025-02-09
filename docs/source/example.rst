Example Usage
=============

To recreate this project the following steps are needed. Data needs to be simulated and stored. This is done by running either main.py or main0403.py. 
The latter, is the version used to generate the dataset used for the final results, therefore it is specially saved. This script will generate particle clusters, 
images as well as .vtk files of them as well as a .csv file with necessary data from the simulation results. After this, either nn_AA.py, nn.py or SVR.py is to be run. 
nn_AA.py will train a neural network as well as the ensemble model using all angles (hence "AA"), for the images of the particle clusters. Furthermore, this file will evaluate 
the model and save plots of the results. The file nn.py will do the same, however, it uses only a single angle for the images of the particle clusters. Finally, the SVR.py 
will train an SVR model and plot results as well as plots.

To summarize:
**Create new dataset**

* Generate new dataset using :meth:`main.Generate_data` and specify the desired simulation parameters, velocity, refinement etc.

**Train new model**

* Run :meth:`nn_AA.main` or :meth:`SVR.SVR_main` with the desired parameters, features and training variables. These scripts will automatically train, test and evalute the performance and plot the result which then can be interpreted. 

**Used saved model to make prediction**

For convinence a EnsembleModel class :meth:`utils.EnsembleModel` is created which structures the different individial networks correctly. This class can load pre-saved weights, train the model on new data or be used to predict drag coefficient on an array of input values or a single input. 
This class have many optional arguements that can be used to specify or modify its behaviour or methods, though the default values are set to what has been tested. It should be plug and play with the weights saved in the data folder, though a potential issue will be the paths when loading. An Example
usage of this class can be seen in :meth:`present_model.exUsageEnsembleModel`
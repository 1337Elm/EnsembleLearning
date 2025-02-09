Main Scripts
============

**These are the main script used for this project:**

.. autofunction:: main.Generate_data

This is the main script for generating a new dataset with a lot of options for particles cluster and simulation configurations

The main variables that can be changed is:

* viscosity - The viscosity of the fluid in the simulation 

* rho - The density of the fluid in the simulation


* Domaindata - Configuration for the simulation domain


* num_particles - Range of number of particles in the cluster that is to be simulated


* vel - Range of velocities in that is to be simulated for **each** particle cluster


* ref - Refinement level around the particle cluster

``Note. Refinement distance and more advanced simulation setting is specified in``:meth:`sphere_3d.create_iboflow_solver`


``Note. The latest version have more advanced settings and generates a larger variety of particle cluster configuration, an older version called main0403 exist to show how the 
dataset from 2024-04-03 was generated!``

In this script there are several calls to helper functions which are explained in :ref:`IBOflow Solver` and :ref:`Utils`

----

.. autofunction:: nn_AA.main

This file contains a main function, it controls the workflow for the machine learning models used. These include, 
the DNN and the MNN which itself includes a NN, CNN and a final NN. The main function starts off by defining dimensions for the networks, 
foldernames and initiating class instances of the different networks. Next up, the :meth:`data_handling.read_csv`, function is called which returns a matrix with all data needed. This in turn, is split up for training, validation and testing of the models. Three different neural netowrks are
initiated from :func:`networks.NeuralNetwork` and :func:`networks.ConvolutionalNeuralNetwork`. Then the respective training algorithms, :meth:`training_testing_AA.training_nn`, :meth:`training_testing_AA.training_cnn` 
and :meth:`training_testing_AA.training_mmnn` are called to train the respective models. Finally, statistics, :meth:`plotting_AA.showcase_model` 
and :meth:`plotting_AA.convergence_plot`,  are called which prints statistical metrics, plots the predictions and the simulated values for the drag coefficient and plots the 
convergence for the models respectively. After this, more plots are created. 

Notable parameters which can easily be changed are all of the network dimensions, given that they are consistent with each other. These are:

* Dataset path

* input_size, hidden_size, output_size

* input_sizeFinal, hidden_sizeFinal, output_sizeFinal

Furthermore, batchsizes are also easily changable. These control how much data the networks analyze in each epoch

* batchsize, batchsize_cnn 

General parameters that are changable are 

* n_epochs - number of maximum epochs 

* tol - tolerance, if loss is less than tol the model is deemed successfull and training is suspended

Furthermore, different loss functions and optimizers can be used. These are placed in the following lists

* loss_fn

* optimizer

These have to be pytorch class instances of the chosen loss function and optimizer respectively.

----

.. autofunction:: SVR.SVR_main

This function loads data from selected dataset and extract the desired features, by default only 2 features (Reynolds' number and projected area is used) but there is no limit of
input features and it can easily be increased to compare. The data is split into 20% test and 80% training, which also easily can be changed as desired. This function used SVR from 
the scikit-learn library and for more information about the workings of SVR please see `scikit-learn-svr <https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVR.html>`_. While 
not strictly needed, all features are scaled before fitting to reduce the impact of outliers and ensure the features are on the same scale. 

Parameters that can be changed:

* Dataset - Path to desired dataset to used

* Cp - Regularization parameter

* eps - Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.

# Tensorflow-Probabillity-
Repo for bayesian neural network.

The file dist_bayesian_nn.py contains the code of distrubuted bayesian neural network for mnist using Horovod.

To run the script execute following on the command line

mpirun -n NOR python3 dist_bayesian_nn.py

where NOR is the number of ranks. 

In case you're running it on one of the XC40 system replace mpirun by aprun in your batch script.

Merge_code folder contains codes for both bayesian nnet and normal nnet (not distributed version).
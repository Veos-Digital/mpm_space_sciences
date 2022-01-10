# Module 1: Introduction

Each module is divided into a theoretical and practical submodule.

**Aims**

1. Gain intuition over fundamental deep learning objects and algorithms.
2. Build a working analysis pipeline.

**Theory.** We introduce definitions of

1. artificial neural networks and multilayer perceptron (MLP).
2. backpropagation: the algorithm used to train the network given an error function, and
3. automatic differentiation.

**Practice.**

1. MLP example: read, use, and complete the functions coded in [mlp_step_by_step.py](./mlp_step_by_step.py) and [activation_function.py](./activation_functions.py). There, the forward and backward functions are implemented line by line and using basic [Torch](https://pytorch.org/docs/stable/index.html) functions.
2. Implement an MLP example by taking advantage of Torch's wrappers and autodifferentiation engine. Test the MLP's performance on the [exoplanet dataset](../../data/exoplanet). For loading the dataset use these [helper functions](../data_management/exoplanet.py).
3. Data are available [here](https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data). Download and save them in a folder named *data* in the repo's main directory. 
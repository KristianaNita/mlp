from mlp.data_providers import CCPPDataProvider
from mlp.layers import AffineLayer
from mlp.errors import SumOfSquaredDiffsError
from mlp.models import SingleLayerModel
from mlp.initialisers import UniformInit, ConstantInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.optimisers import Optimiser
import logging
import numpy as np
import matplotlib.pyplot as plt


# Seed a random number generator
seed = 27092016
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the CCPP training set
train_data = CCPPDataProvider('train', [0, 1], batch_size=100, rng=rng)
input_dim, output_dim = 2, 1

# Create a parameter initialiser which will sample random uniform values
# from [-0.1, 0.1]
param_init = UniformInit(-0.1, 0.1, rng=rng)

# Create our single layer model
layer = AffineLayer(input_dim, output_dim, param_init, param_init)
model = SingleLayerModel(layer)

# Initialise the error object
error = SumOfSquaredDiffsError()

# Use a basic gradient descent learning rule with a small learning rate
learning_rule = GradientDescentLearningRule(learning_rate=1e-2)

# Use the created objects to initialise a new Optimiser instance.
optimiser = Optimiser(model, error, learning_rule, train_data)

# Run the optimiser for 5 epochs (full passes through the training set)
# printing statistics every epoch.
stats, keys = optimiser.train(num_epochs=10, stats_interval=1)

# Plot the change in the error over training.
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
ax.plot(np.arange(1, stats.shape[0] + 1), stats[:, keys['error(train)']])
ax.set_xlabel('Epoch number')
ax.set_ylabel('Error')

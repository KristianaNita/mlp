import matplotlib.pyplot as plt
import numpy as np
import logging
import json
from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import AdamLearningRule
from mlp.optimisers import Optimiser

# plt.style.use('ggplot')


def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, test_data, num_epochs, stats_interval, f=None, notebook=True, early_stopping=False):

    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors = {'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, test_data, data_monitors, notebook=notebook)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time, early_stopping_res = optimiser.train(early_stopping, num_epochs=num_epochs, stats_interval=stats_interval)

    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['error(train)', 'error(valid)']:
        ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval,
                  stats[1:, keys[k]], label=k)
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    for k in ['acc(train)', 'acc(valid)']:
        ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval,
                  stats[1:, keys[k]], label=k)
    ax_2.legend(loc=0)
    ax_2.set_xlabel('Epoch number')
    # plt.show()

    return stats, keys, run_time, early_stopping_res, fig_1, ax_1, fig_2, ax_2


def baseline_early_stopping():
    # Write best epoch number for each combination on file
    with open('./baseline_results/early_stopping_avg_10.txt', 'w') as f:
        i = 0
        for layer in hidden_layers:
            for hidden_unit in relu_hidden_units_per_layer:
                layers = [
                    AffineLayer(input_dim, hidden_unit, weights_init, biases_init),
                    ReluLayer()
                ]

                affine_layer = AffineLayer(hidden_unit, hidden_unit, weights_init, biases_init)
                relu_layer = ReluLayer()

                if layer == 2:
                    layers.append(affine_layer)
                    layers.append(relu_layer)
                if layer == 3:
                    for i in range(2):
                        layers.append(affine_layer)
                        layers.append(relu_layer)

                layers.append(AffineLayer(hidden_unit, output_dim, weights_init, biases_init))
                model = MultipleLayerModel(layers)

                train_model_stats = train_model_and_plot_stats(
                    model, error, learning_rule, train_data, valid_data, test_data, num_epochs, stats_interval, f, notebook=False, early_stopping=True)

                print('\n\n----- ID:', i, ' -----\n\n')
                early_stopping_stats = train_model_stats[3]
                early_stopping_stats['id'] = i
                early_stopping_stats['layers'] = layer
                early_stopping_stats['hidden_units'] = hidden_unit
                f.write('-' * 20 + '\n')
                f.write(json.dumps(early_stopping_stats))
                f.write('\n')
                i += 1


# The below code will set up the data providers, random number
# generator and logger objects needed for training runs. As
# loading the data from file take a little while you generally
# will probably not want to reload the data providers on
# every training run. If you wish to reset their state you
# should instead use the .reset() method of the data providers.

# Seed a random number generator
seed = 11102019
rng = np.random.RandomState(seed)
batch_size = 100
# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
test_data = EMNISTDataProvider('test', batch_size=batch_size, rng=rng)


# The model set up code below is provided as a starting point.
# You will probably want to add further code cells for the
# different experiments you run.

# vsetup hyperparameters
learning_rate = 0.001
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 100

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

# Run baseline for a combination of parameters
hidden_layers = [1, 2, 3, 4, 5]
relu_hidden_units_per_layer = [32, 64, 128]

error = CrossEntropySoftmaxError()
# Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()
baseline_early_stopping()

# Test combinations on test data
# epochs = [30, 23, 10, 19, 12, 9, 12, 11, 8]
# counter = 0
# test_accuracies = dict()

# for layer in hidden_layers:
#     for hidden_unit in relu_hidden_units_per_layer:
#         num_epochs = epochs[counter]
#         print(layer, 'layer,', hidden_unit, 'hidden units\n')

#         stats, keys, run_time, fig_1, ax_1, fig_2, ax_2 = train_model_and_plot_stats(
#             model, error, learning_rule, train_data, test_data, num_epochs, stats_interval, notebook=False)
#         print('test acc:', stats[1:, keys['acc(valid)']])
#         key = str(layer) + '_' + str(hidden_unit)
#         test_accuracies[key] = stats[1:, keys['acc(valid)']]
#         counter += 1

# print('\n')
# print(test_accuracies)

# Write test accuracy to a file
# with open('./baseline_results/test_set_accuracy', 'w+') as test_set_file:
#     for key in test_accuracies:
#         test_set_file.write('' + key + ': ' + str([acc for acc in test_accuracies]))

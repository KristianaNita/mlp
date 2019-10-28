import matplotlib.pyplot as plt
import numpy as np
import logging
import json
import os
from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ParametricReluLayer, RandomReluLayer, ExponentialLinearUnitLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import AdamLearningRule
from mlp.optimisers import Optimiser

# plt.style.use('ggplot')


def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, test_data, num_epochs, stats_interval, early_stopping, notebook=False):

    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors = {'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, test_data, data_monitors, notebook=False)

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


def baseline_early_stopping(early_stopping=True):
    # Write best epoch number for each combination on file
    with open('./baseline_results/early_stopping_new.txt', 'w') as f:
        c_id = 0
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
                    model, error, learning_rule, train_data, valid_data, test_data, num_epochs, stats_interval, early_stopping, notebook=False)

                print('\n\n----- ID:', c_id, ' -----\n\n')
                early_stopping_stats = train_model_stats[3]
                early_stopping_stats['id'] = c_id
                early_stopping_stats['layers'] = layer
                early_stopping_stats['hidden_units'] = hidden_unit
                f.write('-' * 20 + '\n')
                f.write(json.dumps(early_stopping_stats))
                f.write('\n')
                c_id = c_id + 1
                print('\n\n----- ID + 1:', c_id, ' -----\n\n')


def run_experiment(non_linearity, early_stopping, input_dim, hidden_dim, output_dim, alpha=None, filename_path=None, msg=None, write_to_file=False):
    model = MultipleLayerModel([
        AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
        non_linearity() if alpha is None else non_linearity(alpha=alpha),
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
        non_linearity() if alpha is None else non_linearity(alpha=alpha),
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
        non_linearity() if alpha is None else non_linearity(alpha=alpha),
        AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
    ])

    relu_stats = train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, test_data, num_epochs, stats_interval, early_stopping, notebook=False)

    relu_stats = relu_stats[3]

    if write_to_file is True:
        with open(filename_path + '.txt', 'w') as f:
            # f.write('-' * 10 + msg + '-' * 10 + '\n')
            f.write(json.dumps(relu_stats))
            f.write('\n')
            print('File saved at', filename_path)

    return relu_stats


def heatmap(cmap=plt.cm.RdYlGn):
    layers = [1, 2, 3]
    hidden_units = [32, 64, 128]
    # valid_accuracies = np.random.rand(9).reshape(3, 3)
    valid_accuracies = np.array([
        0.7715189873417719,
        0.8124050632911389,
        0.8284177215189871,
        0.779873417721519,
        0.8152531645569618,
        0.8371518987341771,
        0.7760759493670888,
        0.8046202531645571,
        0.8316455696202534
    ]).reshape(3, 3)
    print(valid_accuracies)

    plt.imshow(valid_accuracies, cmap=cmap, interpolation='nearest')
    for i in np.arange(3):
        for j in np.arange(3):
            val = valid_accuracies[j, i]
            # if i == 1:
            plt.text(i, j, '%.3f' % val, ha='center', va='center')

    plt.xticks(np.arange(len(layers)), hidden_units)
    plt.yticks(np.arange(len(hidden_units))[::-1], layers[::-1])
    bottom, top = plt.ylim()
    print(bottom, top)
    plt.ylim([-0.5, 2.5])
    plt.ylabel('Layers')
    plt.xlabel('Hidden units')
    # plt.title("Number of layers and hidden units that maximise the validation accuracy ")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig('./plots/heatmap.pdf', bbox_inches='tight')

    plt.show()


# Linear vs non-linear
# Linear will be the same because if we have 2 matrices: NxK, KxM it's the same as doing NxM.
# Linearity won't transform the inputs
def compare_with_linear():
    model = MultipleLayerModel([
        AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
        AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
    ])

    relu_stats = train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, test_data, num_epochs, stats_interval, True, notebook=False)

    relu_stats = relu_stats[3]

    filename = './experiments/linear/linear_compare'
    with open(filename + '.txt', 'w') as f:
        f.write(json.dumps(relu_stats))
        f.write('\n')
        print('File saved at', filename)


def create_directory(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def run_experiment_with_parameters(params, filename, model, input_dim, hidden_dim, output_dim):
    # filename = './experiments/elu/ELU_alpha_stats.txt'
    with open(filename, 'w+') as f:
        param_dict = dict()

        for param in params:
            print('alpha:', param)
            stats = run_experiment(model, True, input_dim, hidden_dim, output_dim, param)
            param_dict[param] = stats
            f.write(json.dumps(param_dict))
            f.write('\n')

        print(param_dict.keys())
        print('File saved at', filename)


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
hidden_layers = [1, 2, 3]
relu_hidden_units_per_layer = [32, 64, 128]

error = CrossEntropySoftmaxError()
# 2 - Use a basic gradient descent learning rule
learning_rule = AdamLearningRule()
# baseline_early_stopping()

# ------------------------
#
# 3 - Experiments
#
# ------------------------
# hidden_layers = 3
hidden_dim = 128

# Success
# run_experiment(LeakyReluLayer, True, input_dim, hidden_dim, output_dim, './experiments/leaky', 'L-RELU')  # 85.55
# run_experiment(RandomReluLayer, True, input_dim, hidden_dim, output_dim, './experiments/random', 'R-RELU')  # 83.66
# run_experiment(ExponentialLinearUnitLayer, True, input_dim, hidden_dim, output_dim, './experiments/elu', 'ELU')  # 85.48


# Need testing
# run_experiment(ParametricReluLayer, True, input_dim, hidden_dim, output_dim, None, './experiments/parametric/parametric', 'P-RELU', True)

# ------ Random relu -----------
# Test hyperparameters on Random relu
lower = [1 / 10, 1 / 8, 1 / 6]
upper = [1 / 4, 1 / 3, 1 / 2]

# filename = './experiments/random/random_bounds_new.txt'
# with open(filename, 'w+') as f:
#     for l in lower:
#         for u in upper:
#             model = MultipleLayerModel([
#                 AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
#                 RandomReluLayer(lower=l, upper=u),
#                 AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
#                 RandomReluLayer(lower=l, upper=u),
#                 AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
#                 RandomReluLayer(lower=l, upper=u),
#                 AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
#             ])

#             print(model)

#             stats = train_model_and_plot_stats(
#                 model, error, learning_rule, train_data, valid_data, test_data, num_epochs, stats_interval, True, notebook=False)
#             stats = stats[3]

#             f.write('lower ' + str(l) + ', upper ' + str(u))
#             f.write(json.dumps(stats))
#             f.write('\n')
#             print('File saved at', filename)


# --------- Leaky and Parametric ---------
# Test hyperparameters
exponential_alphas = [0.001, 0.01, 0.1, 1, 10]
filename = './experiments/elu/elu_alphas.txt'

# with open(filename, 'w+') as f:
#     for alpha in exponential_alphas:
#         print('alpha', alpha)
#         f.write('alpha ' + str(alpha))
#         stats = run_experiment(ExponentialLinearUnitLayer, True, input_dim, hidden_dim, output_dim, alpha)

#         f.write(json.dumps(stats))
#         f.write('\n')
#     print('File saved at', filename)


# parametric_alphas = [0.1, 0.25, 0.5, 0.75]
# filename = './experiments/parametric/parametric_alphas.txt'

# with open(filename, 'w+') as f:
#     for alpha in parametric_alphas:
#         print('alpha', alpha)
#         f.write('alpha ' + str(alpha))
#         stats = run_experiment(ParametricReluLayer, True, input_dim, hidden_dim, output_dim, alpha)

#         f.write(json.dumps(stats))
#         f.write('\n')
#     print('File saved at', filename)


activation_funcs = ['relu', 'leaky', 'random', 'elu', 'parametric', 'linear']
for func in activation_funcs:
    create_directory('./experiments/' + func)


# Plot each model with default parameters using early stopping
# run_experiment(ReluLayer, True, input_dim, hidden_dim, output_dim, None, './experiments/relu/ReLU_stats', 'RELU', True)
run_experiment(LeakyReluLayer, True, input_dim, hidden_dim, output_dim, None, './experiments/leaky/LReLU_test_stats', 'L-RELU', True)
# run_experiment(RandomReluLayer, True, input_dim, hidden_dim, output_dim, None, './experiments/random/RReLU_stats', 'R-RELU', True)  # 83.66
# run_experiment(ExponentialLinearUnitLayer, True, input_dim, hidden_dim, output_dim, None, './experiments/elu/ELU_stats', 'ELU', True)
# run_experiment(ParametricReluLayer, True, input_dim, hidden_dim, output_dim, None, './experiments/parametric/PReLU_stats', 'PRelu', True)

# @TODO: Plot one model with different alphas
# @TODO: Table with results on test set

# Train with different params
elu_leaky_alphas = [0.001, 0.01, 0.1, 1, 10]
# run_experiment_with_parameters(elu_leaky_alphas, './experiments/leaky/LReLU_alpha_stats.txt', LeakyReluLayer, input_dim, hidden_dim, output_dim)
# run_experiment_with_parameters(elu_leaky_alphas, './experiments/leaky/ELU_alpha_stats.txt', ExponentialLinearUnitLayer, input_dim, hidden_dim, output_dim)

# run_random_relu_with_parameter_bounds()
# lower = [1 / 10, 1 / 8, 1 / 6]
# upper = [1 / 4, 1 / 3, 1 / 2]
# filename = './experiments/random/RReLU_alpha_stats'
# with open(filename, 'w+') as f:
#     param_dict = dict()

#     for l in lower:
#         for u in upper:
#             print('lower:', l)
#             print('upper:', u)
#             model = MultipleLayerModel([
#                 AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
#                 RandomReluLayer(lower=l, upper=u),
#                 AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
#                 RandomReluLayer(lower=l, upper=u),
#                 AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
#                 RandomReluLayer(lower=l, upper=u),
#                 AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
#             ])
#             stats = train_model_and_plot_stats(
#                 model, error, learning_rule, train_data, valid_data, test_data, num_epochs, stats_interval, True, notebook=False)
#             stats = stats[3]
#             full_param = str(l) + '_' + str(u)
#             param_dict[full_param] = stats
#             f.write(json.dumps(param_dict))
#             f.write('\n')

#     print(param_dict.keys())
#     print('File saved at', filename)

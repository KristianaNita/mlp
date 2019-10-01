import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mlp.data_providers import CCPPDataProvider

seed = 27092016
rng = np.random.RandomState(seed)


def fprop(inputs, weights, biases):
    """Forward propagates activations through the layer transformation.

    For inputs `x`, outputs `y`, weights `W` and biases `b` the layer
    corresponds to `y = W x + b`.

    Args:
        inputs: Array of layer inputs of shape (batch_size, input_dim).
        weights: Array of weight parameters of shape
            (output_dim, input_dim).
        biases: Array of bias parameters of shape (output_dim, ).

    Returns:
        outputs: Array of layer outputs of shape (batch_size, output_dim).
    """
    return np.dot(inputs, weights.T) + biases


def error(outputs, targets):
    """Calculates error function given a batch of outputs and targets.

    Args:
        outputs: Array of model outputs of shape (batch_size, output_dim).
        targets: Array of target outputs of shape (batch_size, output_dim).

    Returns:
        Scalar error function value.
    """
    N, dim = outputs.shape
    residuals = outputs - targets
    return np.sum(residuals**2)/(N*2)


def error_grad(outputs, targets):
    """Calculates gradient of error function with respect to model outputs.

    Args:
        outputs: Array of model outputs of shape (batch_size, output_dim).
        targets: Array of target outputs of shape (batch_size, output_dim).

    Returns:
        Gradient of error function with respect to outputs.
        This will be an array of shape (batch_size, output_dim).
    """
    N, dim = outputs.shape
    residuals = outputs - targets
    return residuals/N


def grads_wrt_params(inputs, grads_wrt_outputs):
    """Calculates gradients with respect to model parameters.

    Args:
        inputs: array of inputs to model of shape (batch_size, input_dim)
        grads_wrt_to_outputs: array of gradients of with respect to the model
            outputs of shape (batch_size, output_dim).

    Returns:
        list of arrays of gradients with respect to the model parameters
        `[grads_wrt_weights, grads_wrt_biases]`.
    """

    # grads_wrt_weights = np.sum(np.dot(grads_wrt_outputs, inputs), axis=0)
    grads_wrt_weights = np.dot(inputs.T, grads_wrt_outputs)
    grads_wrt_biases = np.sum(grads_wrt_outputs, axis=0)

    return grads_wrt_weights.T, grads_wrt_biases


inputs = np.array([[0., -1., 2.], [-6., 3., 1.]])
weights = np.array([[2., -3., -1.], [-5., 7., 2.]])
biases = np.array([5., -3.])
true_outputs = np.array([[6., -6.], [-17., 50.]])

if not np.allclose(fprop(inputs, weights, biases), true_outputs):
    print('Wrong outputs computed.')
else:
    print('All outputs correct!')


data_provider = CCPPDataProvider(
    which_set='train',
    input_dims=[0, 1],
    batch_size=5000,
    max_num_batches=1,
    shuffle_order=False
)

input_dim, output_dim = 2, 1

inputs, targets = data_provider.next()

weights_init_range = 0.5
biases_init_range = 0.1

# Randomly initialise weights matrix
weights = rng.uniform(
    low=-weights_init_range,
    high=weights_init_range,
    size=(output_dim, input_dim)
)

#  Randomly initialise biases vector
biases = rng.uniform(
    low=-biases_init_range,
    high=biases_init_range,
    size=output_dim
)
# Calculate predicted model outputs
outputs = fprop(inputs, weights, biases)

# Plot target and predicted outputs against inputs on same axis
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(inputs[:, 0], inputs[:, 1], targets[:, 0], 'r.', ms=2)
ax.plot(inputs[:, 0], inputs[:, 1], outputs[:, 0], 'b.', ms=2)
ax.set_xlabel('Input dim 1')
ax.set_ylabel('Input dim 2')
ax.set_zlabel('Output')
ax.legend(['Targets', 'Predictions'], frameon=False)
fig.tight_layout()

# plt.show()


outputs = np.array([[1., 2.], [-1., 0.], [6., -5.], [-1., 1.]])
targets = np.array([[0., 1.], [3., -2.], [7., -3.], [1., -2.]])
true_error = 5.
true_error_grad = np.array([[0.25, 0.25], [-1., 0.5], [-0.25, -0.5], [-0.5, 0.75]])


if not error(outputs, targets) == true_error:
    print('Error calculated incorrectly.')
elif not np.allclose(error_grad(outputs, targets), true_error_grad):
    print('Error gradient calculated incorrectly.')
else:
    print('Error function and gradient computed correctly!')


inputs = np.array([[1., 2., 3.], [-1., 4., -9.]])
grads_wrt_outputs = np.array([[-1., 1.], [2., -3.]])
true_grads_wrt_weights = np.array([[-3., 6., -21.], [4., -10., 30.]])
true_grads_wrt_biases = np.array([1., -2.])

grads_wrt_weights, grads_wrt_biases = grads_wrt_params(inputs, grads_wrt_outputs)

if not np.allclose(true_grads_wrt_weights, grads_wrt_weights):
    print('Gradients with respect to weights incorrect.')
elif not np.allclose(true_grads_wrt_biases, grads_wrt_biases):
    print('Gradients with respect to biases incorrect.')
else:
    print('All parameter gradients calculated correctly!')

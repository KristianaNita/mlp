import numpy as np
from mlp.test_methods import test_leaky_relu
from mlp.test_methods import test_random_relu
from mlp.test_methods import test_parametric_relu
from mlp.test_methods import test_exponential_linear_unit


# ============
#
# Leaky Relu
#
# ============

fprop_test, fprop_output, fprop_correct, \
bprop_test, bprop_output, bprop_correct = test_leaky_relu()

assert fprop_test == 1.0, (
'The leaky relu fprop functionality test failed'
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(fprop_correct, fprop_output, fprop_output-fprop_correct)
)

print("Leaky ReLU Fprop Functionality Test Passed")

assert bprop_test == 1.0, (
'The leaky relu bprop functionality test failed'
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(bprop_correct, bprop_output, bprop_output-bprop_correct)
)

print("Leaky ReLU Bprop Test Passed")


# ============
#
# Random Relu
#
# ============

fprop_test_arg_leakiness, fprop_output_arg_leakiness, fprop_correct_arg_leakiness, \
bprop_test_arg_leakiness, bprop_output_arg_leakiness, bprop_correct_arg_leakiness, \
fprop_test_rng_leakiness, fprop_output_rng_leakiness, fprop_correct_rng_leakiness, \
bprop_test_rng_leakiness, bprop_output_rng_leakiness, bprop_correct_rng_leakiness = test_random_relu()

assert fprop_test_arg_leakiness == 1.0, (
'The random relu fprop functionality when given an argument leakiness test failed'
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(fprop_correct_arg_leakiness, fprop_output_arg_leakiness, fprop_output_arg_leakiness-fprop_correct_arg_leakiness)
)

print("Random ReLU Fprop when given an argument leakiness Functionality Test Passed")

assert bprop_test_arg_leakiness == 1.0, (
'The random relu bprop functionality when given an argument leakiness test failed'
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(bprop_correct_arg_leakiness, bprop_output_arg_leakiness, bprop_output_arg_leakiness-bprop_correct_arg_leakiness)
)

print("Random ReLU Bprop when using an argument leakiness Functionality Test Passed")

assert fprop_test_rng_leakiness == 1.0, (
'The random relu fprop functionality when given an argument leakiness test failed'
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(fprop_correct_rng_leakiness, fprop_output_rng_leakiness, fprop_output_rng_leakiness-fprop_correct_rng_leakiness)
)

print("Random ReLU Fprop when using rng leakiness Functionality Test Passed")

assert bprop_test_rng_leakiness == 1.0, (
'The random relu bprop functionality when given rng leakiness test failed'
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(bprop_correct_rng_leakiness, bprop_output_rng_leakiness, bprop_output_rng_leakiness-bprop_correct_rng_leakiness)
)

print("Random ReLU Bprop when using rng leakiness Functionality Passed")


# =================
#
# Parametric Relu
#
# =================

fprop_test, fprop_output, fprop_correct, \
bprop_test, bprop_output, bprop_correct, \
grads_wrt_param_test, grads_wrt_param_output, grads_wrt_param_correct = test_parametric_relu()

assert fprop_test == 1.0, (
'The parametric relu fprop functionality test failed'
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(fprop_correct, fprop_output, fprop_output-fprop_correct)
)

print("Parametric ReLU Fprop Functionality Test Passed")

assert grads_wrt_param_test == 1.0, (
'The parametric relu grad_wrt_param functionality test failed'
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(grads_wrt_param_correct, grads_wrt_param_output, grads_wrt_param_output-grads_wrt_param_correct)
)

print("Parametric ReLU Grads wrt Params Test Passed")

assert bprop_test == 1.0, (
'The parametric relu bprop functionality test failed'
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(bprop_correct, bprop_output, bprop_output-bprop_correct)
)

print("Parametric ReLU Bprop Test Passed")


# =================
#
# Exponential Relu
#
# =================

fprop_test, fprop_output, fprop_correct, \
bprop_test, bprop_output, bprop_correct = test_exponential_linear_unit()


assert fprop_test == 1.0, (
'The random relu fprop functionality test failed'
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(fprop_correct, fprop_output, fprop_output-fprop_correct)
)

print("ELU Fprop Functionality Test Passed")

assert bprop_test == 1.0, (
'The random relu bprop functionality test failed'
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(bprop_correct, bprop_output, bprop_output-bprop_correct, bprop_output/bprop_correct)
)

print("ELU Bprop Test Passed")

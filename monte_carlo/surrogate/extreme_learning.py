import numpy as np
from scipy.stats import norm as gauss

from ..util import assure_2d


class FunctionBasis(object):

    def __init__(self, dim):
        self.dim = dim

    def output_matrix(self, xs, params):
        raise NotImplementedError

    def random_params(self, node_count):
        raise NotImplementedError

    def eval_all(self, params, out_bias, out_weights, *xs):
        if len(xs) == 1:
            xs_arr = np.array(xs[0], copy=False, subok=True, ndmin=1)
            xs_arr = assure_2d(xs_arr)
            out = np.dot(self.output_matrix(xs_arr, params), out_weights)
            return out + out_bias
        else:
            first = np.array(xs[0], copy=False, subok=True)
            shape = first.shape
            size = first.size
            xs_arr = np.array(xs, copy=False, subok=True).reshape(len(xs), size)
            xs_arr = assure_2d(xs_arr.transpose())
            out = self.eval_all(params, out_bias, out_weights, xs_arr)
            return out.reshape(shape)

    def eval(self, params, out_bias, out_weights, x):
        """

        :param out_bias: Output bias
        :param params: Parameters about the basis functions (position etc).
        :param out_weights: Numpy array of the individual function weights.
        :param x: Numpy value, position to evaluate function at.
        :return: The approximated function value at the given position.
        """
        xs = np.array(x, copy=False, subok=True, ndmin=2)
        out = np.dot(self.output_matrix(xs, params), out_weights)[0]
        return out + out_bias

    def extreme_learning_train(self, xs, values, node_count, params=None):
        xs = assure_2d(xs)
        if params is None:
            params = self.random_params(node_count)

        out_bias = np.mean(values)

        out_matrix = self.output_matrix(xs, params)
        pinv = np.linalg.pinv(out_matrix)

        weights = np.dot(pinv, values - out_bias)
        return params, out_bias, weights.flatten()


class AdditiveBasis(FunctionBasis):

    def __init__(self, dim, weight_range=(-1, 1), bias_range=(0, -1)):
        """ Linear combination of input followed by non-linear function.

        This class uses the model for additive activation functions from
        "Hamiltonian Monte Carlo acceleration using surrogate
        functions with random bases" (ArXiv ID: 1506.05555):

        If the input is xi and the input weights are wi with biases bi
        (the index i corresponds to the i-th node), the output is
        z(xi) = sum_i gi(wi * xi + bi),
        where the wi and xi are all ndim-dimensional and bi are scalars,
        gi are the activation functions.


        Note that the behaviour is different from Radial Base functions, since
        the position of the "centers" do not only depend on the biases.
        The weights do scale the base function, however it is only applied
        linearly in one dimension.

        Assuming all gi = g are the same and g has a center at 0 (as they would
        be for radial base functions). Then the center for node i would be at
        bi / wi. This behavior can be problematic if g is radial/centered,
        as the biases can not be chosen just based on the x-space region.
        Additionally the bounds for the input biases could not be chosen just
        based on the x-space, thus additive base functions are not a substitute
        for radial base functions.

        :param dim: Dimensionality of variable space (value space is 1D)
        :param weight_range: Range the input weight can have.
        :param bias_range: Range the input bias can have.
        """
        super().__init__(dim)

        self.weight_min = weight_range[0]
        self.weight_delta = weight_range[1] - weight_range[0]

        self.bias_min = bias_range[0]
        self.bias_delta = bias_range[1] - bias_range[0]

    def get_outputs(self, inputs, fn_params):
        raise NotImplementedError

    def random_fn_params(self, node_count):
        raise NotImplementedError

    def output_matrix(self, xs, params):
        biases, in_weights, fn_params = params

        # inputs: node_count * ndim
        inputs = biases[np.newaxis, :] + np.dot(xs, in_weights.transpose())

        outputs = self.get_outputs(inputs, fn_params)
        return outputs

    def random_params(self, node_count):
        biases = self.bias_min + self.bias_delta * np.random.rand(node_count)

        input_weights = np.random.rand(node_count * self.dim)
        input_weights = self.weight_min + self.weight_delta * input_weights
        input_weights = input_weights.reshape(node_count, self.dim)

        return biases, input_weights, self.random_fn_params(node_count)


class RadialBasis(FunctionBasis):

    def __init__(self, dim, width_range=(0, 1), center_range=(0, 1),
                 multi_widths=False):
        """ Linear combination of input followed by non-linear function.

        For a given number of nodes N, use N ndim-dimensional centers ci
        (s.t. -ci is the bias) and N widths wi to approximate
        the output to xi input points (wi are N dimensional if multi_widths is
        True, otherwise 1 dimensional):
        z(xi) = sum_i gi( | (xi + bi) / wi |)
        where gi are activation functions taking 1D inputs.

        :param dim: Dimensionality of variable space (value space is 1D)
        :param width_range: Range the widths can be in. Tuple of either scalars
            or ndim-dimensional numpy arrays (if multi_widths is True).
        :param center_range: Range the centers can be in. Choose such that
            the centers span the x-space. Tuple of either scalars or
            ndim-dimensional numpy arrays.
        :param multi_widths: True if the widths are ndim-dimensional, i.e.
            if the radial functions are stretched independently in each ndim.
        """
        super().__init__(dim)

        self.weight_min = width_range[0]
        self.weight_delta = width_range[1] - width_range[0]

        self.bias_min = center_range[0]
        self.bias_delta = center_range[1] - center_range[0]

        self.multi_width = multi_widths

    def get_outputs(self, inputs, fn_params):
        raise NotImplementedError

    def random_fn_params(self, node_count):
        raise NotImplementedError

    def output_matrix(self, xs, params):
        xs = assure_2d(xs)
        centers, widths, fn_params = params

        # inputs: xs.size * node_count * ndim
        inputs = xs[:, np.newaxis, :] - centers[np.newaxis, :, :]
        if self.multi_width:
            # inputs: (xs.size * node_count * ndim) / (node_count * ndim)
            inputs = inputs / widths
            inputs = np.linalg.norm(inputs, axis=2)
        else:
            inputs = np.linalg.norm(inputs, axis=2)  # take norm
            inputs /= widths

        outputs = self.get_outputs(inputs, fn_params)
        return outputs

    def random_params(self, node_count):
        centers = np.random.random((node_count, self.dim))
        centers = self.bias_min + self.bias_delta * centers

        if self.multi_width:
            widths = np.random.random((node_count, self.dim))
            widths = self.weight_min + self.weight_delta * widths
        else:
            widths = np.random.rand(node_count)
            widths = self.weight_min + self.weight_delta * widths

        return centers, widths, self.random_fn_params(node_count)


# CONCRETE ADDITIVE BASES
class TrigBasis(AdditiveBasis):

    def __init__(self, dim, weight_range=(0, 1), bias_range=(-1, 0)):
        """ Additive function basis using a single Gaussian as non-linearity.

        Example:
        >>> fn = lambda x: np.sin(5 * x)  # want to learn this
        >>> basis = TrigBasis(1)
        >>> xs = np.random.rand(100)      # 100 random points
        >>> values = fn(xs)               # then learn using 100 nodes:
        >>> pars, bias, weights = basis.extreme_learning_train(xs, values, 100)
        >>> val = basis.eval(pars, bias, weights, .4)  # get approx fn value

        :param dim: Dimensionality of variable space (value space is 1D)
        :param weight_range: Range the weight can have.
        :param bias_range: Range the input bias can have.
        """
        super().__init__(dim, weight_range, bias_range)

    def get_outputs(self, inputs, fn_params):
        return np.cos(inputs)

    def random_fn_params(self, node_count):
        return None


# CONCRETE RADIAL BASES
class GaussianBasis(RadialBasis):

    def __init__(self, ndim, width_range=(0, 1), center_range=(0, 1),
                 multi_widths=False):
        """ Radial function basis using a single Gaussian as non-linearity.

        Example:
        >>> fn = lambda x: np.sin(5 * x)  # want to learn this
        >>> basis = GaussianBasis(1)
        >>> xs = np.random.rand(100)      # 100 random points
        >>> values = fn(xs)               # then learn using 100 nodes:
        >>> pars, bias, weights = basis.extreme_learning_train(xs, values, 100)
        >>> val = basis.eval_all(pars, bias, weights, [.1, .2])

        :param ndim: Dimensionality of variable space (value space is 1D)
        :param width_range: Range the widths can be in. Tuple of either scalars
            or ndim-dimensional numpy arrays (if multi_widths is True).
        :param center_range: Range the centers can be in. Choose such that
            the centers span the x-space. Tuple of either scalars or
            ndim-dimensional numpy arrays.
        :param multi_widths: True if the widths are ndim-dimensional, i.e.
            if the radial functions are stretched independently in each ndim.
        """
        super().__init__(ndim, width_range, center_range, multi_widths)

    def get_outputs(self, inputs, fn_params):
        return gauss.pdf(inputs)

    def random_fn_params(self, node_count):
        return None

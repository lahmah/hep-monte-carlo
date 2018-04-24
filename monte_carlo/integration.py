""" Module for Monte Carlo integration methods.

The integral for all techniques is performed over the unit hypercube [0, 1]^ndim.
The integrands must return one dimensional values (e.g. floats)

The module contains several Monte Carlo integration methods, each of which
is wrapped in a class. To create an integrator for certain method, create
an object specifying general settings such as dimensionality or method-specific
parameters. The object is callable with a function (integrand) and iteration
settings to compute the integral. They return the integral estimate and the
(natural estimator of the) standard deviation of that estimate.

Example:
    >>> mc = MonteCarloPlain()  # defaults to one dimensional integration.
    >>> # Integrate fn(x) = x, using 1000 sample points.
    >>> est, err = mc(lambda x: x, 1000)

Advanced variance-reducing Monte Carlo techniques contain several phases,
for each of which the number of iterations and number of function evaluations
need to be specified. Since the structure of the algorithms differ,
the call signature is not the same for all methods.

Example:
    Divide the integration space into 4 equally sized partitions with a base
    number of 10 sample points in each volume.
    >>> volumes = GridVolumes(ndim=1, divisions=4, default_count=10)
    >>> mc_strat = MonteCarloStratified(volumes=volumes)
    >>> # Stratified sampling expects a multiple instead of a total sample size.
    >>> est, err = mc_strat(lambda x: x, 5)  # 5 * 10 sample points per region

To allow sensible comparisons of the different techniques, they all provide
a function for creating an interface where only the total number of function
evaluations (which are considered to be expensive -- the asymptotic efficiency
is usually in this variable) and the integrand are passed.

Example:
    >>> volumes = GridVolumes(ndim=1, divisions=4, default_count=100)
    >>> mc_strat = MonteCarloStratified(volumes=volumes)
    >>> mc_strat_ = mc_strat.get_interface_infer_multiple()
    >>> est, err = mc_strat(lambda x: x, 4000)  # multiple is 4000/(4*100) = 40

The integrand passed to a ndim-dimensional method must take ndim arguments, all
of which are 1d numpy arrays of some size N, and return a numpy array of
the same length N. The numpy vectorize function can help if the integrand of
interest cannot be written using vectors (arrays) easily
(note, however, that vectorize does not increase efficiency).

Example:
    >>> def fn(x, y): return np.sin(x + y)  # x and y are floats
    >>> fn = np.vectorize(fn)                # use @np.vectorize instead
    >>> answer = fn([np.pi, 0], [0, np.pi/2])
    >>> np.allclose(np.array([0, 1]), answer)
    True

    Or more efficiently using addition of arrays, and the fact that np.sin
    accepts and returns numpy arrays:
    >>> def fn(x, y): return np.sin(x + y)  # x and y are numpy arrays
    >>> a, b = np.array([np.pi, 0]), np.array([0, np.pi/2])
    >>> answer = fn(a, b)
    >>> np.allclose(np.array([0, 1]), answer)
    True
"""

import numpy as np
from monte_carlo.sampling import GridVolumes, Channels, assure_2d


# HELPER
def damped_update(old, new, damping_onset, inertia):
    """ Update an old value given a new one, damping the change.

    The parameter inertia can be thought of loosely as the index of the change
    in a number of update iterations, where damping_onset specifies the
    damping-behavior. Both damping_onset and inertia can be floats.

    :param old:
    :param new:
    :param damping_onset: The value of inertia for which old and new values have
        equal weights in the update. Greater values mean smaller damping.
    :param inertia: Value > 0 specifying the strength of damping.
        In the limit to infinity, the old value is retained.
    :return: An updated value combining the old (current) an new values.
    """
    return (new * damping_onset / (inertia + 1 + damping_onset) +  # old
            old * (inertia + 1) / (inertia + 1 + damping_onset))   # new


# MONTE CARLO METHODS
class MonteCarloPlain(object):
    """ Plain Monte Carlo integration method.

    Approximate the integral as the mean of the integrand over a randomly
    selected sample (uniform probability distribution over the unit hypercube).
    """
    def __init__(self, ndim=1, name="MC Plain"):
        self.method_name = name
        self.ndim = ndim

    def __call__(self, fn, eval_count):
        """ Compute Monte Carlo estimate of ndim-dimensional integral of fn.

        The integration volume is the ndim-dimensional unit cube [0,1]^ndim.

        :param fn: A function accepting self.ndim numpy arrays,
            returning an array of the same length with the function values.
        :param eval_count: Total number of function evaluations used to
            approximate the integral.
        :return: Tuple (integral_estimate, error_estimate) where
            the error_estimate is based on the unbiased sample variance
            of the function, computed on the same sample as the integral.
            According to the central limit theorem, error_estimate approximates
            the standard deviation of the statistical (normal) distribution
            of the integral estimates.
        """
        x = np.random.rand(self.ndim * eval_count).reshape(eval_count,
                                                           self.ndim)
        y = fn(*x.transpose())
        mean = np.mean(y)
        var = np.var(y)
        return mean, np.sqrt(var / eval_count)


class MonteCarloImportance(object):

    def __init__(self, ndim=1, sampling=None, pdf=lambda *x: 1,
                 name="MC Importance"):
        """ Importance sampling Monte Carlo integration.

        Importance sampling replaces the uniform sample distribution of plain
        Monte Carlo with a custom pdf.

        By default a uniform probability distribution is used, making the method
        equivalent to plain MC.

        Example:
            >>> sampling = lambda size: np.random.rand(size)**2
            >>> pdf = lambda x: 2*x
            >>> mc_imp = MonteCarloImportance(1, sampling, pdf)
            >>> est, err = mc_imp(lambda x: x, 1000)
            >>> est, err  # the pdf is ideal since fn(x)/(2*x) = 1/2 = const
            (0.5, 0.0)

        :param ndim: Dimensionality of the integral.
        :param sampling: Function taking a number N as argument and returning
            that number of samples distributed according to the pdf.
            It must return a numpy array of shape (N, ndim).
            If ndim=1, sampling may also return an array of shape (N,) instead.
        :param pdf: Distributions of samples used to approximate the integral.
            The function must take ndim numpy arrays of arbitrary but equal
            lengths. The i-th array specifies the coordinates of the
            i-th dimension for all sample points.
        :param name: Name of the method that can be used as label in
            plotting routines (can be changed to name parameters).
        """
        self.method_name = name
        self.ndim = ndim

        if sampling is None:
            # default is a uniform distribution
            def sampling(sample_size):
                """ Uniform sampling of sample_size ndim-dimensional values. """
                sample = np.random.rand(sample_size * self.ndim)
                return sample.reshape(sample_size, self.ndim)

        self.sampling = sampling
        self.pdf = pdf

    def __call__(self, fn, eval_count):
        """ Approximate the integral of fn.

        :param fn: Integrand, taking ndim numpy arrays and returning a number.
        :param eval_count: Total number of function evaluations.
        :return: Tuple (integral_estimate, error_estimate).
        """
        x = assure_2d(self.sampling(eval_count))

        y = fn(*x.transpose()) / self.pdf(*x.transpose())

        mean = np.mean(y)  # integral estimate
        var = np.var(y)    # variance of the weighted function samples
        return mean, np.sqrt(var / eval_count)


class MonteCarloMultiImportance(object):

    def __init__(self, channels: Channels, b=.5, name="MC Multi C.",
                 var_weighted=False):
        """ Multi channel Monte Carlo integration.

        Use multiple importance sampling channels to approximate the integral.
        Each of the channels has a weight that is adapted and optimized
        throughout the integration.

        The integration is split into three phases, each of which containing
        several iterations (of which each contains several function
        evaluations).

        Phase 1: Update weights but discard integral estimates.
        Phase 2: Update weights and use the results for the integral estimation.
        Phase 3: Don't update weights and continue using the current weights to
            estimate the integral.

        Example:
            >>> uniform_pdf = lambda *x: 1
            >>> uniform_sampling = lambda size: np.random.rand(size)
            >>> channels = Channels([uniform_sampling], [uniform_pdf])
            >>> mc_imp = MonteCarloMultiImportance(channels)  # same as plain MC
            >>> est, err = mc_imp(lambda x: x, [], [100], [])

        :param channels: (Importance sampling) Channels used in the integration.
        :param b: Exponent between 1/2 and 1/4, used in updating the weights.
        :param name: Name of the method used for plotting.
        :param var_weighted: If true, weight the estimates from different
            iterations with their variance (to obtain the best estimate).
            Note that this can lead to a bias if the variances and estimates
            are correlated.
        """
        self.method_name = name
        self.channels = channels
        self.var_weighted = var_weighted
        self.b = b

    def get_interface_ratios(self, sub_eval_count=100, r1=0, r2=1, r3=0):
        """ Get an interface to the integration that only takes a sample size.

        Fix the ratios of function evaluations spent in each phase and infer
        the multiple of a fixed number sub_eval_count of function evaluations
        for each phase.

        If a given eval_count cannot be split equally into bins of size
        sub_eval_count, the remaining evaluations are spent in phase 3.

        :param sub_eval_count: Number of function evaluations in
            each iteration.
        :param r1: Gives the ratio of iterations spent in the first phase.
        :param r2: Ratio of second phase (by default 1-r1-r3).
        :param r3: Ratio of third phase (by default 0).
        :return: A function serving as interface.
        """
        if r1 < 0 or r2 < 0 or r3 < 0:
            raise ValueError(
                "Ratios cannot be smaller than 0: %d, %d, %d." % (r1, r2, r3))
        if not np.isclose(1, r1 + r2 + r3):
            raise ValueError("Ratios must sum to 1: %d + %d + %d = %d." % (
                r1, r2, r3, r1 + r2 + r3))

        def interface(fn, eval_count, apriori=True):
            """ Approximate the integral of fn via using given sample size.

            The method used to approximate the integral is Multi channel
            Monte Carlo.

            :param fn: Integrand.
            :param eval_count: Total number of function evaluations.
            :param apriori: If true, reset the channel weights for each
                call to the method.
            :return: Tuple (integral_estimate, error_estimate).
            """
            num_iterations = eval_count // sub_eval_count
            m1 = int(r1 * num_iterations)
            m2 = int(r2 * num_iterations)
            m3 = int(r3 * num_iterations)
            samples_remaining = eval_count - (m1 + m2 + m3) * sub_eval_count

            iter_1 = [sub_eval_count] * m1
            iter_2 = [sub_eval_count] * m2
            iter_3 = [sub_eval_count] * m3
            if samples_remaining:
                iter_3.append(samples_remaining)
            return self(fn, iter_1, iter_2, iter_3, apriori=apriori)

        interface.method_name = self.method_name

        return interface

    def iterate(self, fn, eval_count, update_weights=True, get_estimate=True):
        """ One iteration of the algorithm with sample size eval_count.

        :param fn: Integrand.
        :param eval_count: Number of function evaluations in the iteration.
        :param update_weights: If true, channel weights are updated according
            to the respective contributions to the variance.
        :param get_estimate: Specify if integral estimate (i.e. function mean)
            and sample variance should be computed.
        :return: If compute_estimate is true, estimate and contribution to
            sample variance of the estimate w_est. Otherwise return nothing.
            The variance of the estimate is (w_est - est^2) / eval_count
        """
        # a ChannelSample object
        sample = self.channels.generate_sample(eval_count)

        # weighted samples of fn
        fn_samples = (fn(*sample.sample.transpose()) / sample.sample_weights)
        # contribution to variance of fn
        w_fn = (np.add.reduceat(fn_samples ** 2, sample.channel_bounds) /
                sample.count_per_channel)

        if update_weights:
            factors = sample.channel_weights * np.power(w_fn, self.b)
            self.channels.update_channel_weights(factors / np.sum(factors))

        if get_estimate:
            estimate = np.mean(fn_samples)
            w_est = np.sum(sample.channel_weights * w_fn)
            return estimate, w_est

    def __call__(self, fn, eval_count_1, eval_count_2, eval_count_3,
                 apriori=True):
        """ Approximate the integral of fn over the [0,1]^ndim hypercube.

        :param fn: Integrand.
        :param eval_count_1: List of sample sizes for the iterations
            in phase 1.
        :param eval_count_2: List of sample sizes for the iterations
            in phase 2.
        :param eval_count_3: List of sample sizes for the iterations
            in phase 3.
        :param apriori: If true, reset the channel weights.
        :return: Tuple (integral_estimate, error_estimate).
        """
        if apriori:
            self.channels.reset()

        for eval_count in eval_count_1:
            self.iterate(fn, eval_count, True, False)

        eval_counts = np.concatenate([eval_count_2, eval_count_3])
        eval_counts = eval_counts.astype(np.int)
        m2 = len(eval_count_2)  # number of iterations in phase 2
        ws_est = np.empty(eval_counts.size)
        estimates = np.empty(eval_counts.size)

        for j, eval_count in zip(range(eval_counts.size), eval_counts):
            estimates[j], ws_est[j] = self.iterate(fn, eval_count, j < m2, True)

        if self.var_weighted:
            # sample variance of individual iterations
            variances = (ws_est - estimates ** 2) / eval_counts
            norm = np.sum(eval_counts / variances)
            total_est = np.sum(eval_counts * estimates / variances) / norm
            var = np.sum(eval_counts ** 2 / variances) / norm ** 2
        else:
            total_evaluations = np.sum(eval_counts)
            total_est = np.sum(estimates * eval_counts) / total_evaluations
            var = (np.sum(eval_counts * ws_est / total_evaluations) -
                   total_est ** 2) / total_evaluations

        return total_est, np.sqrt(var)


# STRATIFIED
class MonteCarloStratified(object):

    def __init__(self, volumes=GridVolumes(), name="MC Stratified"):
        """ Stratified Monte Carlo integration.

        Approximate N-dimensional integral of fn over the unit hypercube
        by sampling the function independently in a specified division of
        the integration volume.

        :param volumes: Partition of the unit hypercube volume.
            If using a Volumes class other than GridVolumes, it must implement
            a method iterate that behaves as the one in GridVolumes.
        :param name: Name of the method used for plotting.
        """
        self.method_name = name
        self.ndim = volumes.ndim
        self.volumes = volumes

    def get_interface_infer_multiple(self):
        """ Construct an interface that only takes fn and a total sample size.

        If the sample size is not an integer multiple of
        self.volumes.total_base_size, the actual number of function evaluations
        might be lower than N.
        """
        def interface(fn, eval_count):
            """ Approximate the integral of fn via using given sample size.

            The method used to approximate the integral is stratified MC.

            :param fn: Integrand.
            :param eval_count: Total number of function evaluations.
            :return: Tuple (integral_estimate, error_estimate).
            """
            return self(fn, eval_count / self.volumes.total_base_size)

        interface.method_name = self.method_name

        return interface

    def __call__(self, fn, multiple):
        """ Approximate the integral of fn using stratified sampling MC.

        :param fn: Integrand.
        :param multiple: Multiply the base sample size of each region with this
            number. The total number of function evaluations will then be
            multiple * self.volumes.total_base_size.
        :return: Tuple (integral_estimate, error_estimate).
        """
        int_est = 0
        var_est = 0
        for sub_eval_count, sample, vol in self.volumes.iterate(multiple):
            values = fn(*sample.transpose())
            int_est += vol * np.mean(values)
            var_est += np.var(values) * vol ** 2 / sub_eval_count
        return int_est, np.sqrt(var_est)


# VEGAS
class MonteCarloVEGAS(object):

    def __init__(self, ndim=1, divisions=1, c=3,
                 name="MC VEGAS", var_weighted=False):
        """ VEGAS Monte Carlo integration algorithm.


        :param ndim: Dimensionality of the integral.
        :param divisions: Number of divisions of the volume along
            each dimension. The total number of 'boxes' is divisions^ndim
        :param c: Damping parameter (greater implies less damping), used in
            adapting the division sizes. See damped_update of this module.
        :param name: Method name used for plotting.
        :param var_weighted: If true, weight the estimates from different
            iterations with their variance (to obtain the best estimate).
            Note that this can lead to a bias if the variances and estimates
            are correlated.
        """
        # the configuration is defined by the sizes of the bins
        self.sizes = np.ones((ndim, divisions)) / divisions
        self.ndim = ndim
        self.volumes = GridVolumes(ndim=ndim, divisions=divisions)
        # number of bins along each axis
        self.divisions = divisions
        self.c = c
        self.var_weighted = var_weighted
        self.method_name = name

    def get_interface_infer_multiple(self, sub_eval_count):
        """ Construct an interface that only takes fn and a total sample size.

        If the sample size is not an integer multiple of sub_eval_count,
        the actual number of function evaluations might be lower than N.

        :param sub_eval_count: Number of function evaluations per iteration.
        """
        def interface(fn, eval_count, apriori=True, chi=False):
            """ Approximate the integral of fn via using given sample size.

            The method used to approximate the integral is the VEGAS algorithm.

            :param chi: If true return the chi^2/dof value of the estimates
                obtained in the various iterations.
            :param apriori: If true, reset the division sizes.
            :param fn: Integrand.
            :param eval_count: Total number of function evaluations.
            :return: Tuple (integral_estimate, error_estimate).
            """
            # inevitably do fewer evaluations for some eval_count values
            iterations = eval_count // sub_eval_count
            return self(fn, sub_eval_count, iterations, apriori, chi)

        interface.method_name = self.method_name

        return interface

    def choice(self):
        """ Return the multi-index of a random bin. """
        return np.random.randint(0, self.divisions, self.ndim)

    def pdf(self, indices):
        """ Probability density of finding a point in bin with given index. """
        # since here N = 1 = const for all bins,
        # the pdf is given simply by the volume of the regions
        vol = np.prod([self.sizes[d][indices[d]] for d in range(self.ndim)],
                      axis=0)
        return 1 / self.divisions ** self.ndim / vol

    def plot_pdf(self):
        """ Plot the pdf resulting from the current bin sizes. """
        self.volumes.plot_pdf(label="VEGAS pdf")

    def __call__(self, fn, sub_eval_count, iterations, apriori=True, chi=False):
        """ Approximate the integral of fn using stratified sampling.

        :param fn: Integrand.
        :param sub_eval_count: Number of function evaluations per iteration.
        :param iterations: Number of iterations.
        :param apriori: If true, reset the sizes of the division.
        :param chi: Indicate whether chi^2 over the estimates of all
            iterations should be computed. Can only do this if iterations >= 2.
        :return: If chi=False tuple (integral_estimate, error_estimate),
            otherwise the tuple (integral_estimate, error_estimate, chi^2/dof).
        """
        if apriori:
            # start anew
            self.sizes = np.ones((self.ndim, self.divisions)) / self.divisions
            self.volumes.update_bounds_from_sizes(self.sizes)

        assert not chi or iterations > 1, "Can only compute chi^2 if there" \
                                          "is more than one iteration "

        est_j = np.zeros(iterations)  # The estimate in each iteration j
        var_j = np.zeros(iterations)  # sample estimate of the variance of est_j
        # keep track of contributions of each marginalized bin to the estimate
        estimates = np.zeros((self.ndim, self.divisions))

        for j in range(iterations):
            indices = self.volumes.random_bins(sub_eval_count)
            samples_t = self.volumes.sample(indices)
            values = fn(*samples_t) / self.pdf(indices)
            for div in range(self.divisions):
                estimates[:, div] = np.mean(values * (indices == div), axis=1)
            est_j[j] = np.mean(values)
            var_j[j] = np.var(values) / sub_eval_count

            inv_sizes = damped_update(est_j[j] / self.sizes,
                                      self.divisions * estimates, self.c, j)
            self.sizes = 1 / inv_sizes
            # normalize
            self.sizes /= np.add.reduce(self.sizes, axis=1)[:, np.newaxis]
            self.volumes.update_bounds_from_sizes(self.sizes)

        # note: weighting with sub_eval_count here is redundant,
        # but illustrates how this algorithm could be expanded
        # (could modify to make sub_eval_count vary with j)

        if self.var_weighted:
            norm = np.sum(sub_eval_count / var_j)  # normalization factor
            total_est = np.sum(sub_eval_count * est_j / var_j) / norm
            # final estimate: weight by sub_eval_count and var_j
            var = np.sum(sub_eval_count ** 2 / var_j) / norm ** 2
        else:
            total_est = (np.sum(sub_eval_count * est_j) /
                         (iterations * sub_eval_count))
            var = (np.sum(sub_eval_count ** 2 * var_j) /
                   (iterations * sub_eval_count) ** 2)

        if chi:
            # chi^2/dof, have "iteration" values that are combined,
            # so here dof = iterations - 1
            chi2 = np.sum((est_j - total_est) ** 2 / var_j) / (iterations - 1)
            return total_est, np.sqrt(var), chi2

        return total_est, np.sqrt(var)

import numpy as np
from typing import Tuple

from ..sampling import Sample
from ..density import Density


class PlainMC(object):
    """ Plain Monte Carlo integration method.

    Approximate the integral as the mean of the integrand over a randomly
    selected sample (uniform probability distribution over the unit hypercube).
    """
    def __init__(self, ndim: int = 1, name: str = "MC Plain") -> None:
        self.method_name = name
        self.ndim = ndim

    def __call__(self, target: Density, eval_count: int) -> Tuple[float, float]:
        """ Compute Monte Carlo estimate of ndim-dimensional integral of fn.

        The integration volume is the ndim-dimensional unit cube [0,1]^ndim.

        Parameters
        ----------
        target
            The target density.
        eval_count
            Total number of function evaluations used to
            approximate the integral.

        Returns
        -------
        Tuple[float, float]
            Tuple (integral_estimate, error_estimate) where
            the error_estimate is based on the unbiased sample variance
            of the function, computed on the same sample as the integral.
            According to the central limit theorem, error_estimate approximates
            the standard deviation of the statistical (normal) distribution
            of the integral estimates.
        """
        data = np.random.random((eval_count, self.ndim))
        function_values = target.pdf(data)
        integral = np.mean(function_values)
        err = np.sqrt(np.var(function_values) / eval_count)
        return integral, err

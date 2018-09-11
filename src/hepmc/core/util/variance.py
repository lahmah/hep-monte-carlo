
# numerically stable computation of variance 
# samples are added incrementally and variance can be calculated in the end
# for online calculation use online_variance
class incremental_variance:
	def __init__(self):
		self.K = 0.
		self.n = 0
		self.Ex = 0.
		self.Ex2 = 0.

	def add_variable(self,x):
		if self.n is 0:
			self.K = x

		self.n += 1
		self.Ex += x-self.K
		self.Ex2 += (x-self.K)*(x-self.K)

	
	def get_meanvalue(self):
		return self.K + self.Ex/self.n

	def get_variance(self):
		return (self.Ex2 - (self.Ex*self.Ex)/self.n) / (self.n-1)

class online_variance:
    """Numerically stable online calculation of variance.

    Based on Welford's algorithm.

    Attributes
    ----------
    self.n : int
        Number of entries.
    self.mean : float
        Mean of all entries.
    self.M2 : float
        Variable used to calculate the variance.
    
    .. todo::
        Use hidden attributes and make actual getters.
    """
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.
        self.M2 = 0.

    def add_variable(self, x: float) -> None:
        """Add an entry.

        Parameters
        ----------
        x
            The entry to add.
        """
        self.n += 1
        delta = x-self.mean
        self.mean += delta/self.n
        delta2 = x - self.mean
        self.M2 += delta*delta2

    def get_mean(self) -> float:
        """Get the mean of all entries.
        
        Returns
        -------
        float
            The mean of all entries.
        """
        return self.mean

    def get_variance(self) -> float:
        """Get the variance of all entries.

        Returns
        -------
        float
            The variance of all entries.
        """
        return self.M2/(self.n-1)

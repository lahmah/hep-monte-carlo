import numpy as np

from ..density import Density
from .base import MarkovUpdate
from ..proposals import Gaussian


class MetropolisState(np.ndarray):
    def __new__(cls, input_array, pot=None):
        obj = np.array(input_array, copy=False, subok=True, ndmin=1).view(cls)
        if pot is not None:
            obj.pot = pot
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return  # was called from the __new__ above
        self.pot = getattr(obj, 'pot', None)


# METROPOLIS (HASTING) UPDATE
class MetropolisUpdate(MarkovUpdate):

    def __init__(self, target, adaptive=False, hasting=False):
        """ Generic abstract class to represent a single Metropolis-like update.
        """
        super().__init__(target, is_adaptive=adaptive)
        self.is_hasting = hasting

    def adapt(self, iteration, prev, state, accept):
        pass

    def accept(self, state, candidate):
        """ Default accept implementation for Metropolis/Hasting update.

        :param state: Previous state in the Markov chain.
        :param candidate: Candidate for next state.
        :return: The acceptance probability of a candidate state given the
            previous state in the Markov chain.
        """
        if self.is_hasting:
            return state.pot + self.proposal_mlogpdf(state, candidate) - candidate.pot - self.proposal_mlogpdf(candidate, state)
        else:
            # otherwise (simpler) Metropolis update
            return state.pot - candidate.pot

    def proposal(self, state):
        """ A proposal generator.

        Generate candidate points in the sample space.
        These are used in the update mechanism and
        accepted with a probability self.accept(candidate) that depends
        on the used algorithm.

        :param state: The previous state in the Markov chain.
        :return: A candidate state of type MetropolisState with pdf set.
        """
        raise NotImplementedError("MetropolisLikeUpdate is abstract.")

    def proposal_pdf(self, state, candidate):
        pass  # Implement for Hasting update.

    def proposal_mlogpdf(self, state, candidate):
        pass  # Implement for Hasting update.

    def init_state(self, state):
        if not isinstance(state, MetropolisState):
            state = MetropolisState(state)
        if state.pot is None:
            state.pot = self.target.pot(state)

        return super().init_state(state)

    def next_state(self, state, iteration):
        candidate = self.proposal(state)

        try:
            accept = self.accept(state, candidate)
        except (TypeError, AttributeError):
            # in situations like mixing/composite updates, previous update
            # may not have set necessary attributes (such as pdf)
            state = self.init_state(state)
            accept = self.accept(state, candidate)

        if np.log(np.random.rand()) < accept:
            next_state = candidate
        else:
            next_state = state

        if self.is_adaptive:
            self.adapt(iteration, state, next_state, accept)

        return next_state


class DefaultMetropolis(MetropolisUpdate):

    def __init__(self, target, proposal=None, cov=None, adaptive=False):
        """ Use the Metropolis algorithm to generate a sample.

        Example:
            >>> pdf = lambda x: np.sin(10*x)**2
            >>> met = DefaultMetropolis(1, pdf)    # 1 dimensional
            >>> sample = met.sample(1000, 0.1)   # generate 1000 samples

        :param ndim: Dimensionality of sample space.
        :param target: Desired (unnormalized) probability distribution.
            Either a function accepting a numpy array of shape (ndim,) or
            a Density object.
        :param proposal: A Proposal object.
        """
        if proposal is None:
            proposal = Gaussian(target.ndim, cov)
        self._proposal = proposal

        # must be at the and since it calls proposal_pdf to see if it works
        super().__init__(target,
                         adaptive=adaptive, hasting=not proposal.is_symmetric)

    def proposal(self, state):
        #candidate = self._proposal.proposal(np.asarray(state))
        candidate = self._proposal.proposal(state)
        return MetropolisState(candidate, self.target.pot(candidate))

    def proposal_pdf(self, state, candidate):
        return self._proposal.proposal_pdf(state, candidate)

    # minus log pdf
    def proposal_mlogpdf(self, state, candidate):
        return self._proposal.proposal_mlogpdf(state, candidate)

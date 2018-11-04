import numpy as np
from collections import deque
from ..sampling import Sample
from ..util import is_power_of_ten
from ..density import Density
from tqdm import tqdm

# MARKOV CHAIN
class MarkovUpdate(object):
    """Basic update mechanism of a Markov chain. """

    def __init__(self, target: Density, is_adaptive: bool = False) -> None:
        self.target = target
        self.is_adaptive = is_adaptive

        # will hold information if update was used as a sampler
        self.sample_info = None

    def init_adapt(self, initial_state):
        pass

    def init_state(self, state):
        """Sets the needed attributes (e.g. the potential) of a state.

        This abstract implementation does nothing.
        """
        return state

    def next_state(self, state, iteration: int):
        """Get the next state in the Markov chain.

        Returns
        -------
        MarkovState
            The next state.
        """
        raise NotImplementedError("AbstractMarkovUpdate is abstract.")

    def generator(self, sample_size: int, init_state, lag=1):
        """Returns a generator that yields new states sequentially."""
        i = 0
        state = init_state
        while i < sample_size:
            for j in range(lag):
                state = self.next_state(state, i)
            yield state
            i += 1

    def sample(self, sample_size: int, init_state, burnin: int = 0, batch_length=100, lag=1) -> Sample:
        init_state = self.init_state(init_state)
        self.init_adapt(init_state)  # initial adaptation
        batch_accept = deque(maxlen=batch_length)
        previous = init_state

        if burnin > 0:
            with tqdm(total=burnin, desc="Burn-in (lag=1)") as pbar:
                for i, state in enumerate(self.generator(burnin, init_state)):
                    if not np.array_equal(state, previous):
                        batch_accept.append(1)
                    else:
                        batch_accept.append(0)

                    if ((i+1) % batch_length) == 0:
                        accept_rate = sum(batch_accept)/batch_length
                        pbar.set_postfix({"batch acc. rate" : accept_rate})

                        pbar.update(batch_length)

                    previous = state
        init_state = state

        data = np.empty((sample_size, self.target.ndim))
        with tqdm(total=sample_size, desc='Sampling (lag={})'.format(lag)) as pbar:
            for i, state in enumerate(self.generator(sample_size, init_state, lag)):
                data[i] = state

                if not np.array_equal(state, previous):
                    batch_accept.append(1)
                else:
                    batch_accept.append(0)

                if ((i+1) % batch_length) == 0:
                    accept_rate = sum(batch_accept)/batch_length
                    pbar.set_postfix({"batch acc. rate" : accept_rate})

                    pbar.update(batch_length)

                previous = state

        return Sample(data=data, target=self.target)

class CompositeMarkovUpdate(MarkovUpdate):

    def __init__(self, ndim, updates, masks=None, target=None):
        """ Composite Markov update; combine updates.

        :param updates: List of update mechanisms, each subtypes of
            MetropolisLikeUpdate.
        :param masks: Dictionary, giving masks (list/array of indices)
            of dimensions for the index of the update mechanism. Use this if
            some updates only affect slices of the state.
        """
        is_adaptive = any(update.is_adaptive for update in updates)
        if target is None:
            for update in updates:
                if update.target is not None:
                    target = update.target
                    break
        super().__init__(ndim, is_adaptive=is_adaptive, target=target)

        self.updates = updates
        self.masks = [None if masks is None or i not in masks else masks[i]
                      for i in range(len(updates))]

    def init_adapt(self, initial_state):
        for update in self.updates:
            state = update.init_state(initial_state)
            update.init_adapt(state)

    def next_state(self, state, iteration):
        for mechanism, mask in zip(self.updates, self.masks):
            if mask is None:
                state = mechanism.next_state(state, iteration)
            else:
                state = np.copy(state)
                state[mask] = mechanism.next_state(state[mask], iteration)

        return state


class MixingMarkovUpdate(MarkovUpdate):

    def __init__(self, updates, weights=None, masks=None,
                 in_maps=None, out_maps=None, target=None):
        """ Mix a number of update mechanisms, choosing one in each step.

        :param updates: List of update mechanisms (AbstractMarkovUpdate).
        :param weights: List of weights for each of the mechanisms (sum to 1).
        :param masks: Slice object, specify if updates only affect slice of
            state.
        """
        is_adaptive = any(update.is_adaptive for update in updates)
        if target is None:
            for update in updates:
                if update.target is not None:
                    target = update.target
                    break
        super().__init__(is_adaptive=is_adaptive, target=target)

        self.updates = updates
        self.updates_count = len(updates)
        self.masks = [None if masks is None or i not in masks else masks[i]
                      for i in range(len(updates))]
        if weights is None:
            weights = np.ones(self.updates_count) / self.updates_count
        self.weights = weights

        self.in_maps = in_maps or dict()
        self.out_maps = out_maps or dict()

    def init_adapt(self, initial_state):
        for i, update in enumerate(self.updates):
            try:
                init = self.in_maps[i](initial_state)
            except KeyError:
                init = initial_state
            state = update.init_state(init)
            update.init_adapt(state)

    def next_state(self, state, iteration):
        update_index = np.random.choice(self.updates_count, p=self.weights)
        update = self.updates[update_index]
        try:
            state = self.in_maps[update_index](state).flatten()
        except KeyError:
            pass

        if self.masks[update_index] is None:
            state = update.init_state(state)
            next_state = update.next_state(state, iteration)
        else:
            mask = self.masks[update_index]
            state = np.copy(state)
            state[mask] = update.next_state(state[mask], iteration)
            next_state = state

        try:
            return self.out_maps[update_index](next_state).flatten()
        except KeyError:
            return next_state

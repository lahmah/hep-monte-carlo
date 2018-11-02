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

    #def sample(self, sample_size: int, initial, out_mask=None, n_batches: int = 20) -> Sample:
    #    """Generate a sample of given size.

    #    Parameters
    #    ----------
    #    sample_size
    #        Number of samples to generate.
    #    initial
    #        Initial state of the Markov chain. Can be a MarkovState
    #        or a numpy array.
    #    out_mask
    #        Slice object, return only this slice of the output
    #        chain (useful if sampler uses artificial variables).

    #    Returns
    #    -------
    #    Sample
    #        A sample object containing the data.
    #    """
    #    # initialize sampling
    #    state = self.init_state(np.atleast_1d(initial))
    #    if len(state) != self.target.ndim:
    #        raise ValueError('initial must have dimension ' + str(self.target.ndim))
    #    self.init_adapt(state)  # initial adaptation

    #    batch_length = int(sample_size/n_batches)

    #    tags = dict()
    #    tagged = dict()

    #    chain = np.empty((sample_size, self.target.ndim))
    #    chain[0] = state

    #    batch_accept = deque(maxlen=batch_length)
    #    current_seq = 1 # current sequence length
    #    max_seq = 1 # maximal sequence length
    #    skip = 1
    #    for i in range(1, sample_size):
    #        state = self.next_state(state, i)
    #        if not np.array_equal(state, chain[i - 1]):
    #            batch_accept.append(1)
    #            if current_seq > max_seq:
    #                max_seq = current_seq
    #            current_seq = 1
    #        else:
    #            batch_accept.append(0)
    #            current_seq += 1

    #        chain[i] = state
    #        try:
    #            try:
    #                tags[state.tag_parser].append(state.tag)
    #                tagged[state.tag_parser].append(i)
    #            except KeyError:
    #                tags[state.tag_parser] = []
    #                tagged[state.tag_parser] = []
    #        except AttributeError:
    #            pass

    #        if i % skip == 0:
    #            if i >= batch_length:
    #                accept_rate = sum(batch_accept)/batch_length
    #            else:
    #                accept_rate = sum(batch_accept)/i
    #            if i == 1:
    #                print("Event 1\t(batch acceptance rate: %f)" % (accept_rate))
    #            else:
    #                print("Event %i\t(batch acceptance rate: %f)\tmax sequence length: %i" % (i, accept_rate, max(current_seq, max_seq)))
    #            if is_power_of_ten(i):
    #                skip *= 10

    #    if out_mask is not None:
    #        chain = chain[:, out_mask]

    #    for parser in tagged:
    #        chain[tagged[parser]] = parser(chain[tagged[parser]], tags[parser])

    #    sample = Sample(data = chain, target = self.target)
    #    return sample

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

    def __init__(self, ndim, updates, weights=None, masks=None,
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
        self.ndim = ndim

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

    def next_state(self, state, iteration, update_index):
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

    def sample(self, sample_size, initial, out_mask=None, n_batches=20):
        """ Generate a sample of given size.

        :param sample_size: Number of samples to generate.
        :param initial: Initial value of the Markov chain. Internally
            converted to numpy array.
        :param out_mask: Slice object, return only this slice of the output
            chain (useful if sampler uses artificial variables).
        :return: Numpy array with shape (sample_size, self.ndim).
        """
        # initialize sampling
        state = self.init_state(np.atleast_1d(initial))
        if len(state) != self.ndim:
            raise ValueError('initial must have dimension ' + str(self.ndim))
        self.init_adapt(state)  # initial adaptation

        batch_length = int(sample_size/n_batches)

        tags = dict()
        tagged = dict()

        # produce a new initial state from a known update 
        index = np.random.choice(self.updates_count, p=self.weights)
        state = self.next_state(state, 0, index)

        chain = np.empty((sample_size, self.ndim))
        weights = np.empty(sample_size)
        weight_owner = np.empty(sample_size, dtype=np.uint8)
        chain[0] = state
        try:
            weights[0] = state.weight
        except AttributeError:
            # if the update is weightless, we set the weight to 1 and normalize it later
            weights[0] = 1
        weight_owner[0] = index

        batch_accept = deque(maxlen=batch_length)
        current_seq = 1 # current sequence length
        max_seq = 1 # maximal sequence length
        skip = 1
        for i in range(1, sample_size):
            index = np.random.choice(self.updates_count, p=self.weights)
            state = self.next_state(state, i, index)
            if not np.array_equal(state, chain[i - 1]):
                batch_accept.append(1)
                if current_seq > max_seq:
                    max_seq = current_seq
                current_seq = 1
            else:
                batch_accept.append(0)
                current_seq += 1

            chain[i] = state
            try:
                weights[i] = state.weight
            except AttributeError:
                # if the update is weightless, we set the weight to 1 and normalize it later
                weights[i] = 1
            weight_owner[i] = index

            try:
                try:
                    tags[state.tag_parser].append(state.tag)
                    tagged[state.tag_parser].append(i)
                except KeyError:
                    tags[state.tag_parser] = []
                    tagged[state.tag_parser] = []
            except AttributeError:
                pass

            if i % skip == 0:
                if i >= batch_length:
                    accept_rate = sum(batch_accept)/batch_length
                else:
                    accept_rate = sum(batch_accept)/i
                if i == 1:
                    print("Event 1\t(batch acceptance rate: %f)" % (accept_rate))
                else:
                    print("Event %i\t(batch acceptance rate: %f)\tmax sequence length: %i" % (i, accept_rate, max(current_seq, max_seq)))
                if is_power_of_ten(i):
                    skip *= 10

        if out_mask is not None:
            chain = chain[:, out_mask]

        for parser in tagged:
            chain[tagged[parser]] = parser(chain[tagged[parser]], tags[parser])

        # normalize weights
        for i in range(self.updates_count):
            is_owned = weight_owner == i
            mask = np.where(is_owned)
            count = np.sum(is_owned)
            sum_weights = np.sum(weights[mask])
            weights[mask] = weights[mask] / sum_weights * (count/sample_size)

        sample = Sample(data=chain, target=self.target, weights=weights)
        return sample

import numpy as np
from collections import deque
from .hmc import HamiltonianUpdate, HamiltonState
from ..densities.gaussian import Gaussian
from ..sampling import Sample
from ..util import is_power_of_ten


def integrator(q, pot_gradient, v, du, lim_lower, lim_upper, stepsize, nsteps):
    # make a half step for velocity
    v = v - stepsize/2 * du
    
    # alternate full steps for position and momentum
    for l in range(nsteps):
        # make a full step for position
        q = q + stepsize * v

        # handle constraints by wall hitting
        while(True):
            l_c = q < lim_lower
            u_c = q > lim_upper

            if l_c.any():
                q[l_c] = 2*lim_lower[l_c[0]] - q[l_c]
                v[l_c] = -v[l_c]
            elif u_c.any():
                q[u_c] = 2*lim_upper[u_c[0]] - q[u_c]
                v[u_c] = -v[u_c]
            else:
                break

        # make full step for velocity
        du = pot_gradient(q)[0]
        if np.isinf(du).any():
            return None, None, None            

        if l != nsteps-1:
            v = v - stepsize * du

    # make last half step for velocity
    v = v - stepsize/2 * du
    
    return q, v, du


class WallHMC(HamiltonianUpdate):
    """
    Wall HMC for box type constraints
    """

    def __init__(self, target_density, stepsize_min, stepsize_max,
                 nsteps_min, nsteps_max, lim_lower=None, lim_upper=None,
                 is_adaptive=False):
        p_dist = Gaussian(target_density.ndim)
        super().__init__(target_density, p_dist, None, None, is_adaptive=is_adaptive)
        self.target_density = target_density
        self.p_dist = p_dist

        # default limits: unit hypercube
        if lim_lower is None:
            lim_lower = np.zeros(self.target_density.ndim)
        if lim_upper is None:
            lim_upper = np.ones(self.target_density.ndim)

        self.stepsize_min = stepsize_min
        self.stepsize_max = stepsize_max
        self.nsteps_min = nsteps_min
        self.nsteps_max = nsteps_max
        self.lim_lower = lim_lower
        self.lim_upper = lim_upper

    def init_state(self, state):
        if not isinstance(state, HamiltonState):
            state = HamiltonState(state)

        return super().init_state(state)
    
    def proposal(self, current):
        """Propose a new state."""
        # initialization
        #try:
        #    q = current
        #    du = current.pot_gradient
        #except AttributeError:
        #    current = self.init_state(current)
        #    q = current
        #    du = current.pot_gradient
        q = current
        du = self.target_density.pot_gradient(q)

        # sample velocity
        v = current.momentum = self.p_dist.proposal()

        # sample integrator parameters
        nsteps = np.random.randint(self.nsteps_min, self.nsteps_max + 1)
        stepsize = (np.random.rand() * (self.stepsize_max - self.stepsize_min) +
                    self.stepsize_min)
        
        # integrate
        q, v, du = integrator(q, self.target_density.pot_gradient, v, du,
                              self.lim_lower, self.lim_upper, stepsize, nsteps)

        if q is None:
            return None

        #pot = self.target_density.pot(q)
        #return HamiltonState(q, momentum=v, pot_gradient=du, pot=pot)
        return HamiltonState(q, momentum=v)

    def accept(self, state, candidate):
        """Return the logarithm of the acceptance probability."""
        try:
            U_current = self.target_density.pot(state)
            #if np.isinf(U_current): # shouldn't be necessary
            #    return 0
            H_current = U_current + .5*state.momentum.dot(state.momentum)
            U_proposal = self.target_density.pot(candidate)
            if np.isinf(U_proposal):
                return -np.inf
            H_proposal = U_proposal + .5*candidate.momentum[0].dot(candidate.momentum[0])
            log_prob = -H_proposal + H_current
            if np.isinf(log_prob): # shouldn't be necessary
                return -np.inf
            return log_prob
        except RuntimeWarning:
            return -np.inf

    def next_state(self, state, iteration):
        candidate = self.proposal(state)
        if candidate is None:
            return state

        try:
            log_accept = self.accept(state, candidate)
        except (TypeError, AttributeError):
            # in situations like mixing/composite updates, previous update
            # may not have set necessary attributes (such as pdf)
            state = self.init_state(state)
            log_accept = self.accept(state, candidate)

        #if not np.isinf(log_accept) and np.log(np.random.rand()) < min(0, log_accept):
        if np.log(np.random.rand()) < log_accept:
            next_state = candidate
        else:
            next_state = state

        if self.is_adaptive:
            self.adapt(iteration, state, next_state, log_accept)

        return next_state

    def sample(self, sample_size, initial, out_mask=None, n_batches=20):
        """
        Return a weighted sample. To get an unweighted sample it has 
        to be resampled using np.random.choice()
        """

        # initialize sampling
        state = self.init_state(np.atleast_1d(initial))
        if len(state) != self.target_density.ndim:
            raise ValueError('initial must have dimension ' + str(self.target_density.ndim))
        self.init_adapt(state)  # initial adaptation

        batch_length = int(sample_size/n_batches)

        tags = dict()
        tagged = dict()

        chain = np.empty((sample_size, self.target_density.ndim))
        chain[0] = state

        batch_accept = deque(maxlen=batch_length)
        current_seq = 1 # current sequence length
        max_seq = 1 # maximal sequence length
        skip = 1
        for i in range(1, sample_size):
            state = self.next_state(state, i)
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

        sample = Sample(data=chain, target=self.target_density)
        return sample

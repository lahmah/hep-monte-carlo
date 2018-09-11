import numpy as np
from collections import deque
from .hmc import HamiltonianUpdate, HamiltonState
from ..core.densities import Gaussian
from ..core.markov import MarkovSample
from ..core.util import is_power_of_ten


class SphericalHMCState(HamiltonState):

    #@staticmethod
    #def tag_parser(chain, log_weights):
    #    log_weights = np.asanyarray(log_weights)
    #    weights = np.exp(log_weights - log_weights.mean())
    #    weights /= weights.sum()
    #    count = len(chain)
    #    sub = np.random.choice(np.arange(count), count,
    #                           replace=True, p=weights)
    #    return chain[sub]

    def __new__(cls, input_array, theta=None, tag=None, pot_gradient=None, weight=None,
                **kwargs):
        obj = super().__new__(cls, input_array, **kwargs)
        if theta is not None:
            obj.theta = theta
        obj.tag = tag  # tag is reset
        if pot_gradient is not None:
            obj.pot_gradient = pot_gradient
        obj.weight = weight
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return  # was called from the __new__ above
        super().__array_finalize__(obj)
        self.theta = getattr(obj, 'theta', None)
        self.tag = getattr(obj, 'tag', None)
        self.pot_gradient = getattr(obj, 'pot_gradient', None)


def cot(x):
    if (x == 0).any():
        return np.inf
    else:
        return 1/np.tan(x)


def arccot(x):
    return np.pi/2 - np.arctan(x)


def integrator(q, pot_gradient, v, du, trafo, jac, stepsize, nsteps):
    cumsinq = np.cumprod(np.sin(q))

    # make a half step for velocity
    v = v - stepsize/2 * du/np.concatenate(([1], cumsinq[:-1]**2))
    
    # alternate full steps for position and momentum
    for l in range(nsteps):
        # make a full step for position
        # 1. map to augmented sphere
        x = np.concatenate((np.cos(q), [1])) * np.concatenate(([1], cumsinq))
        dx = (np.concatenate((-v*np.tan(q), [0])) +
              np.concatenate(([0], np.cumsum(v*cot(q))))) * x
        # 2. rotate on sphere
        x0 = x
        dx_nom = np.sqrt(np.sum(dx**2))
        costdx = np.cos(dx_nom*stepsize)
        sintdx = np.sin(dx_nom*stepsize)
        x = x0*costdx + dx/dx_nom*sintdx
        dx = -x0*dx_nom*sintdx + dx*costdx
        # 3. go back to hyper-rectangle
        cumx2 = np.cumsum(x**2)
        cotq = x[:-1] / np.sqrt(1-cumx2[:-1])
        q = arccot(cotq)
        q[-1] = np.pi + np.sign(x[-1])*(q[-1]-np.pi)
        v = -cotq*(dx[:-1]/x[:-1]+np.concatenate(([0], np.cumsum(x*dx)[:-2])) /
                   (1-np.concatenate(([0], cumx2[:-2]))))
        v[-1] = v[-1] * np.sign(x[-1])
        
        # make last half step for velocity
        cumsinq = np.cumprod(np.sin(q))
        du = pot_gradient(trafo(q))[0]*jac
        if l != nsteps-1:
            v = v - stepsize * du/np.concatenate(([1], cumsinq[:-1]**2))
    
    return q, v, du


class StaticSphericalHMC(HamiltonianUpdate):
    """
    Spherical HMC in spherical coordinates with unit radius
    for box type constraints
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
            lim_lower = np.zeros(self.ndim)
        if lim_upper is None:
            lim_upper = np.ones(self.ndim)

        self.stepsize_min = stepsize_min
        self.stepsize_max = stepsize_max
        self.nsteps_min = nsteps_min
        self.nsteps_max = nsteps_max
        self.lim_lower = lim_lower
        self.lim_upper = lim_upper
        
        self.J_xtheta = (self.lim_upper-self.lim_lower)/(np.concatenate(
            (np.full(self.ndim-1, 1), [2]))*np.pi)

    def theta_to_x(self, theta):
        return self.lim_lower + theta*self.J_xtheta
    
    def x_to_theta(self, x):
        return (x-self.lim_lower) / self.J_xtheta

    def init_state(self, state):
        if not isinstance(state, SphericalHMCState):
            state = SphericalHMCState(state, theta=self.x_to_theta(state))
        if state.theta is None:
            state.theta = self.x_to_theta(np.array(state, copy=False))
        if state.tag is None:
            state.tag = self.log_weight(state.theta)
        if state.pot_gradient is None:
            state.pot_gradient = self.target_density.pot_gradient(state)[0]
        if state.weight is None:
            state.weight = np.exp(self.log_weight(state.theta))

        return super().init_state(state)
    
    def proposal(self, current):
        """Propose a new state in the spherical domain."""
        # initialization
        try:
            q = current.theta
            du = current.pot_gradient
        except AttributeError:
            current = self.init_state(current)
            q = current.theta
            du = current.pot_gradient

        # sample velocity
        z = current.momentum = self.p_dist.proposal()
        cumsinq = np.cumprod(np.sin(q))
        v = z / np.concatenate(([1], cumsinq[:-1]))

        ## evaluate potential and kinetic energy at current state
        #U = self.target_density.pot(current)
        #H_current = U + .5*z.dot(z)
        
        # sample integrator parameters
        nsteps = np.random.randint(self.nsteps_min, self.nsteps_max + 1)
        stepsize = (np.random.rand() * (self.stepsize_max - self.stepsize_min) +
                    self.stepsize_min)
        
        # integrate
        q, v, du = integrator(q, self.target_density.pot_gradient, v, du,
                              self.theta_to_x, self.J_xtheta, stepsize, nsteps)

        z = (v*np.concatenate(([1], cumsinq[:-1])) -
             stepsize/2 * du/np.concatenate(([1], cumsinq[:-1])))
    
        ## evaluate potential and kinetic energy at current state
        #U = self.target_density.pot(q)
        #H_proposal = U + .5*z.dot(z)
        
        x = self.theta_to_x(q)
        prob = self.target_density.pdf(x)
        #return SphericalHMCState(x, momentum=z, tag=self.log_weight(q),
        #                         pot_gradient=du, pdf=prob, theta=q)
        return SphericalHMCState(x, momentum=z, pot_gradient=du, pdf=prob, theta=q)

    def accept(self, state, candidate):
        """Return the logarithm of the acceptance probability."""
        try:
            #prob = (candidate.pdf * self.p_dist.pdf(candidate.momentum) /
            #        state.pdf / self.p_dist.pdf(state.momentum))
            U_current = self.target_density.pot(state)
            H_current = U_current + .5*state.momentum.dot(state.momentum)
            U_proposal = self.target_density.pot(candidate)
            H_proposal = U_proposal + .5*candidate.momentum.dot(candidate.momentum)
            log_prob = -H_proposal + H_current
            if np.isinf(log_prob):
                return 0
            return log_prob
        except RuntimeWarning:
            return 0

    def next_state(self, state, iteration):
        candidate = self.proposal(state)

        try:
            log_accept = self.accept(state, candidate)
        except (TypeError, AttributeError):
            # in situations like mixing/composite updates, previous update
            # may not have set necessary attributes (such as pdf)
            state = self.init_state(state)
            log_accept = self.accept(state, candidate)

        if np.log(np.random.rand()) < min(0, log_accept):
            next_state = candidate
        else:
            next_state = state

        # give the state a weight
        next_state.weight = np.exp(self.log_weight(next_state.theta))

        if self.is_adaptive:
            self.adapt(iteration, state, next_state, log_accept)

        return next_state

    def log_weight(self, q):
        log_weight = (np.arange(1, self.ndim) - self.ndim).dot(
            np.log(np.sin(q[:-1]))) + np.sum(np.log(self.J_xtheta))
        return log_weight

    def sample(self, sample_size, initial, out_mask=None, n_batches=20):
        """
        Return a weighted sample. To get an unweighted sample it has 
        to be resampled using np.random.choice()
        """

        # initialize sampling
        state = self.init_state(np.atleast_1d(initial))
        if len(state) != self.ndim:
            raise ValueError('initial must have dimension ' + str(self.ndim))
        self.init_adapt(state)  # initial adaptation

        batch_length = int(sample_size/n_batches)

        sample = MarkovSample()

        tags = dict()
        tagged = dict()

        chain = np.empty((sample_size, self.ndim))
        chain[0] = state
        #log_weights = np.empty(sample_size)
        #log_weights[0] = self.log_weight(state)
        weights = np.empty(sample_size)
        weights[0] = state.weight

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
                sample.accepted += 1
            else:
                batch_accept.append(0)
                current_seq += 1

            chain[i] = state
            #log_weights[i] = self.log_weight(state)
            weights[i] = state.weight
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

        sample.data = chain
        #sample.weights = np.exp(log_weights - log_weights.mean())
        sample.weights = weights
        sample.target = self.target_density
        return sample


class DualAveragingSphericalHMC(StaticSphericalHMC):
    """
    Adapts the stepsize by dual averaging
    """
    
    def __init__(self, target_density, simulation_length,
                 adapt_schedule, lim_lower=None, lim_upper=None, t0=10,
                 stepsize_bar0=1, Hbar0=0, gamma=0.05, kappa=0.75, delta=0.65):
        super().__init__(target_density, None, None, None, None,
                         lim_lower, lim_upper, True)
        
        self.simulation_length = simulation_length
        self.adapt_schedule = adapt_schedule
        self.t0 = t0
        self.stepsize_bar = stepsize_bar0
        self.Hbar = Hbar0
        self.gamma = gamma
        self.kappa = kappa
        self.delta = delta

        # set in init_state
        self.mu = None
        # replace with step_size after refactoring integrate
        self.stepsize = None

    def init_adapt(self, state):
        state = self.init_state(state)
        super().init_adapt(state)
        # as of now, do not use self.step_size
        self.stepsize = self.find_reasonable_step_size(state)
        self.stepsize_min = self.stepsize_max = self.stepsize
        # print('stepsize:', self.step_size)
        self.nsteps_min = int(self.simulation_length / self.stepsize)
        self.nsteps_max = self.nsteps_min
        self.mu = np.log(10 * self.stepsize)
        
    def find_reasonable_step_size(self, current):
        stepsize = 1.
        
        # initialization
        current_u = self.target_density.pot(current)
        current_du = current.pot_gradient
        
        # sample velocity
        current_z = np.random.standard_normal(self.ndim)
        
        # evaluate energy at start of trajectory
        E_cur = current_u + .5*np.sum(current_z**2)
        
        # integrate one step
        proposal_q, proposal_z, proposal_du = integrator(
            current.theta, self.target_density.pot_gradient, current_z,
            current_du, self.theta_to_x, self.J_xtheta, stepsize, nsteps=1)
        
        # evaluate energy at the end of the trajectory
        proposal_u = self.target_density.pot(self.theta_to_x(proposal_q))
        E_prp = proposal_u + .5*np.sum(proposal_z**2)

        # aprob = np.exp(-E_cur + E_prp)
        #aprob = DualAveragingSphericalHMC.accept(
        #    self,
        #    SphericalHMCState(None, momentum=current_z, pdf=self.target_density.pdf(current)),
        #    SphericalHMCState(None, momentum=proposal_z, pdf=self.target_density.pdf(self.theta_to_x(proposal_q))))
        aprob = np.exp(-E_prp+E_cur)
        # print('aprob:', aprob)
            
        a = 2. * (aprob > 0.5) - 1.
        # print('a', a)
        while aprob**a > 2**(-a):
            stepsize = 2.**a * stepsize
            # print('stepsize:', stepsize)
            proposal_q, proposal_z, proposal_du = integrator(
                current.theta, self.target_density.pot_gradient, current_z,
                proposal_du, self.theta_to_x, self.J_xtheta, stepsize, nsteps=1)
            proposal_u = self.target_density.pot(self.theta_to_x(proposal_q))
            E_prp = proposal_u + .5*np.sum(proposal_z**2)
            aprob = np.exp(-E_cur + E_prp)
            #aprob = DualAveragingSphericalHMC.accept(
            #    self,
            #    SphericalHMCState(None, momentum=current_z,
            #                      pdf=self.target_density.pdf(current)),
            #    SphericalHMCState(None, momentum=proposal_z,
            #                      pdf=self.target_density.pdf(
            #                          self.theta_to_x(proposal_q))))

            # print('aprob:', aprob)

        # limit the stepsize to reasonable values
        min_stepsize = 1e-3
        return max(stepsize, min_stepsize)
    
    def adapt(self, iteration, prev, current, log_accept):
        accept = np.exp(log_accept)
        if self.adapt_schedule(iteration) is True:
            self.Hbar = (1 - 1 / (iteration + self.t0)) * \
                        self.Hbar + 1 / (iteration + self.t0) * \
                        (self.delta - accept)
            log_stepsize = self.mu - np.sqrt(iteration) / self.gamma * self.Hbar
            self.stepsize = np.exp(log_stepsize)
            if self.stepsize >= self.simulation_length:
                self.stepsize = self.simulation_length
                log_stepsize = np.log(self.stepsize)

            self.stepsize_bar = np.exp(iteration ** (-self.kappa) * log_stepsize
                                       + (1 - iteration ** (-self.kappa)) *
                                       np.log(self.stepsize_bar))
            self.nsteps_min = self.nsteps_max = int(self.simulation_length /
                                                    self.stepsize)
        
        else:
            self.stepsize = self.stepsize_bar
            if self.stepsize >= self.simulation_length:
                self.stepsize = self.simulation_length
                
            self.nsteps_min = self.nsteps_max = int(self.simulation_length /
                                                    self.stepsize)
        
        #print('aprob:', aprob)
        #print('stepsize:', self.stepsize)
        #print('nsteps:', self.nsteps_min)

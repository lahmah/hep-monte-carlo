import numpy as np
from scipy.optimize import brentq


MINKOWSKI = np.diag([1, -1, -1, -1])


def map_fourvector_rambo(xs):
    """ Transform unit hypercube points into into four-vectors. """
    c = 2. * xs[:, :, 0] - 1.
    phi = 2. * np.pi * xs[:, :, 1]

    q = np.empty_like(xs)
    q[:, :, 0] = -np.log(xs[:, :, 2] * xs[:, :, 3])
    q[:, :, 1] = q[:, :, 0] * np.sqrt(1 - c ** 2) * np.cos(phi)
    q[:, :, 2] = q[:, :, 0] * np.sqrt(1 - c ** 2) * np.sin(phi)
    q[:, :, 3] = q[:, :, 0] * c

    return q


def map_rambo(xs, E_CM, nparticles=None):
    if nparticles is None:
        nparticles = xs.shape[1] // 4

    p = np.empty((xs.shape[0], nparticles, 4))

    q = map_fourvector_rambo(xs.reshape(xs.shape[0], nparticles, 4))
    # sum over all particles
    Q = np.add.reduce(q, axis=1)

    M = np.sqrt(np.einsum('kd,dd,kd->k', Q, MINKOWSKI, Q))
    b = (-Q[:, 1:] / M[:, np.newaxis])
    x = E_CM / M
    gamma = Q[:, 0] / M
    a = 1. / (1. + gamma)

    bdotq = np.einsum('ki,kpi->kp', b, q[:, :, 1:])

    # make dimensions match
    gamma = gamma[:, np.newaxis]
    x = x[:, np.newaxis]
    p[:, :, 0] = x * (gamma * q[:, :, 0] + bdotq)

    # make dimensions match
    b = b[:, np.newaxis, :]  # dimensions: samples * nparticles * space dim)
    bdotq = bdotq[:, :, np.newaxis]
    x = x[:, :, np.newaxis]
    a = a[:, np.newaxis, np.newaxis]
    p[:, :, 1:] = x * (q[:, :, 1:] + b * q[:, :, 0, np.newaxis] + a * bdotq * b)

    return p.reshape(xs.shape)

def two_body_decay_factor(M_i_minus_1, M_i, m_i_minus_1):
    return 1./(8*M_i_minus_1**2) * np.sqrt((M_i_minus_1**2 - (M_i+m_i_minus_1)**2)*(M_i_minus_1**2 - (M_i-m_i_minus_1)**2))

def boost(q, ph):
    p = np.empty(q.shape)

    rsq = np.sqrt(np.einsum('kd,dd,kd->k', q, MINKOWSKI, q))

    p[:, 0] = np.einsum('ki,ki->k', q, ph) / rsq
    c1 = (ph[:, 0]+p[:, 0]) / (rsq+q[:, 0])
    p[:, 1:] = ph[:, 1:] + c1[:, np.newaxis]*q[:, 1:]

    return p

def map_rambo_on_diet(xs, E_CM, nparticles=None):
    if nparticles is None:
        nparticles = (xs.shape[1] + 4) // 3

    p = np.empty((xs.shape[0], nparticles, 4))

    #q = np.empty((xs.shape[0], 4))
    M = np.zeros((xs.shape[0], nparticles))
    u = np.empty((xs.shape[0], nparticles-2))

    Q = np.tile([E_CM, 0, 0, 0], (xs.shape[0], 1))
    Q_prev = np.empty((xs.shape[0], 4))
    M[:, 0] = E_CM

    for i in range(2, nparticles+1):
        Q_prev[:, :] = Q[:, :]
        if i != nparticles:
            u[:, i-2] = [brentq(lambda x : (nparticles+1-i)*x**(2*(nparticles-i)) - (nparticles-i)*x**(2*(nparticles+1-i)) - r_i, 0., 1.) for r_i in xs[:, i-2]]
            M[:, i-1] = np.product(u[:, :i-1], axis=1)*E_CM
        
        cos_theta = 2*xs[:, nparticles-6+2*i] -1
        phi = 2*np.pi*xs[:, nparticles-5+2*i]
        q = 4*M[:, i-2]*two_body_decay_factor(M[:, i-2], M[:, i-1], 0)

        p[:, i-2, 0] = q
        p[:, i-2, 1] = q*np.cos(phi)*np.sqrt(1-cos_theta**2)
        p[:, i-2, 2] = q*np.sin(phi)*np.sqrt(1-cos_theta**2)
        p[:, i-2, 3] = q*cos_theta
        Q[:, 0] = np.sqrt(q**2+M[:, i-1]**2)
        Q[:, 1:] = -p[:, i-2, 1:]
        p[:, i-2] = boost(Q_prev, p[:, i-2])
        Q = boost(Q_prev, Q)

    p[:, nparticles-1] = Q

    return p.reshape((xs.shape[0], nparticles*4))

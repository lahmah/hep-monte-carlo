#!/usr/bin/python3

from monte_carlo import hamiltonian, densities
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)

ndim = 2
bananicity = 0.1
target = densities.Banana(ndim, bananicity)

nsamples = 100
#start = np.array([0., 10.])
start = np.array([-20., -30.])

stepsize = .3
nsteps = 50
proposal = densities.Gaussian(ndim)
sampler = hamiltonian.HamiltonianUpdate(target, proposal, nsteps, stepsize)

sample = sampler.sample(nsamples, start)

n_accepted = 1
for i in range(1, nsamples):
    if (sample.data[i] != sample.data[i-1]).any():
            n_accepted += 1

print('Acceptance rate:', n_accepted/nsamples)


a = np.sqrt(599)
b = np.sqrt(5.99)
x = np.linspace(-a, a, 200)
ellipse = b/a*np.sqrt(a**2-x**2)
contour1 = ellipse - bananicity*x**2 + 100*bananicity
contour2 = -ellipse - bananicity*x**2 + 100*bananicity

plt.figure(1)
plt.plot(x, contour1, 'b-', zorder=1)
plt.plot(x, contour2, 'b-', zorder=1)
plt.scatter(sample.data[:, 0], sample.data[:, 1], 3, 'red', zorder=2)
plt.xlim([-40, 40])
plt.ylim([-60, 20])

plt.axes().set_aspect('equal')
plt.savefig('hmc_banana.pdf', dpi=300, bbox_inches='tight')
plt.show()


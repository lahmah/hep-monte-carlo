import sys
import Sherpa
from ..density import Density
from ..util import interpret_array
import pkg_resources
import numpy as np
from itertools import combinations

# e+ e- -> q qbar + n gluons
class ee_qq_ng(Density):
    def __init__(self, n_gluons, E_CM, pT_min, angle_min):
        self.conversion = 0.389379*1e9  # convert to picobarn
        self.nfinal = 2+n_gluons  # number of final state particles
        ndim = 4 * self.nfinal

        super().__init__(ndim, False)

        self.n_gluons = n_gluons
        self.E_CM = E_CM
        self.pT_min = pT_min
        self.angle_min = angle_min

        self.Generator = Sherpa.Sherpa()
        self.Generator.InitializeTheRun(4,
                                        [''.encode('ascii'),
                                         ('RUNDATA=' + path_to_runcard(n_gluons)).encode('ascii'),
                                         'INIT_ONLY=2'.encode('ascii'),
                                         'OUTPUT=0'.encode('ascii')])
        self.Process = Sherpa.MEProcess(self.Generator)

        # Incoming flavors must be added first!
        self.Process.AddInFlav(11)
        self.Process.AddInFlav(-11)
        self.Process.AddOutFlav(1)
        self.Process.AddOutFlav(-1)
        for _ in range(n_gluons):
            self.Process.AddOutFlav(21)
        self.Process.Initialize()

        self.Process.SetMomentum(0, E_CM/2., 0., 0., E_CM/2.)
        self.Process.SetMomentum(1, E_CM/2., 0., 0., -E_CM/2.)

    # The first momentum is xs[0:4]
    # The second momentum is xs[4:8], ...
    def pdf(self, xs):
        xs = interpret_array(xs, self.ndim)

        ndim = xs.shape[1]

        if ndim != self.ndim:
            raise RuntimeWarning("Mismatching dimensions.")

        sample_size = len(xs)
        me = np.empty(sample_size)

        cross_section = np.empty(xs.shape[0])
        for i in range(sample_size):
            momenta = []
            for j in range(self.nfinal):
                momenta.append(xs[i, 4*j:4*j+4])

            # apply cuts
            if all([pT(p)>self.pT_min for p in momenta]) and all([angle(p1, p2)>self.angle_min for p1, p2 in combinations(momenta, 2)]):
                for j, momentum in enumerate(momenta):
                    self.Process.SetMomentum(j+2, momentum[0], momentum[1], momentum[2], momentum[3])

                me[i] = self.Process.CSMatrixElement()
            else:
                me[i] = 0.

            cross_section = (2. * np.pi) ** (4.-3. * self.nfinal) / (2. * self.E_CM ** 2) * me

        return self.conversion * cross_section

def path_to_runcard(n_gluons):
    return pkg_resources.resource_filename('hepmc', 'data/ee_qq_' + str(n_gluons) + 'g.dat')

# Calculate transverse momentum of 4-vector
def pT(p):
    return np.sqrt(p[1]*p[1]+p[2]*p[2])

# Calculate angle between two 4-vectors
def angle(p, q):
    cos_angle = (p[1:].dot(q[1:]))/(p[0]*q[0])
    return np.arccos(cos_angle)

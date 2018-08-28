import sys
import Sherpa
from ..density import Density
from ..util import interpret_array
import pkg_resources
import numpy as np
from itertools import combinations


CARD0 = pkg_resources.resource_filename('hepmc', 'data/ee_qq.dat')
CARD1 = pkg_resources.resource_filename('hepmc', 'data/ee_qq_1g.dat')
CARD2 = pkg_resources.resource_filename('hepmc', 'data/ee_qq_2g.dat')


# e+ e- -> q qbar
class ee_qq(Density):
    def __init__(self, E_CM):
        ndim = 8
        self.conversion = 0.389379*1e9  # convert to picobarn
        self.nfinal = 2  # number of final state particles

        super().__init__(ndim, False)

        self.E_CM = E_CM

        self.Generator = Sherpa.Sherpa()
        self.Generator.InitializeTheRun(3,
                                        [''.encode('ascii'),
                                         ('RUNDATA=' + CARD0).encode('ascii'),
                                         'INIT_ONLY=2'.encode('ascii'),
                                         'OUTPUT=0'.encode('ascii')])
        self.Process = Sherpa.MEProcess(self.Generator)

        # Incoming flavors must be added first!
        self.Process.AddInFlav(11)
        self.Process.AddInFlav(-11)
        self.Process.AddOutFlav(1)
        self.Process.AddOutFlav(-1)
        self.Process.Initialize()

        self.Process.SetMomentum(0, E_CM/2., 0., 0., E_CM/2.)
        self.Process.SetMomentum(1, E_CM/2., 0., 0., -E_CM/2.)

    # The first momentum is xs[0:4]
    # The second momentum is xs[4:8]
    def pdf(self, xs):
        xs = interpret_array(xs, self.ndim)

        ndim = xs.shape[1]

        if ndim != self.ndim:
            raise RuntimeWarning("Mismatching dimensions.")

        p1 = xs[:, 0:4]
        p2 = xs[:, 4:8]

        sample_size = len(xs)
        me = np.empty(sample_size)
        for i in range(sample_size):
            self.Process.SetMomentum(2, p1[i, 0], p1[i, 1], p1[i, 2], p1[i, 3])
            self.Process.SetMomentum(3, p2[i, 0], p2[i, 1], p2[i, 2], p2[i, 3])
            me[i] = self.Process.CSMatrixElement()

        xs = (2. * np.pi) ** (4.-3. * self.nfinal) / (2. * self.E_CM ** 2) * me
        return self.conversion * xs


# e+ e- -> q qbar + gluon
class ee_qq_1g(Density):
    def __init__(self, E_CM, pT_min, angle_min):
        ndim = 4 * 3
        self.conversion = 0.389379*1e9  # convert to picobarn
        self.nfinal = 3  # number of final state particles

        super().__init__(ndim, False)

        self.E_CM = E_CM
        self.pT_min = pT_min
        self.angle_min = angle_min

        self.Generator = Sherpa.Sherpa()
        self.Generator.InitializeTheRun(4,
                                        [''.encode('ascii'),
                                         ('RUNDATA=' + CARD1).encode('ascii'),
                                         'INIT_ONLY=2'.encode('ascii'),
                                         'OUTPUT=0'.encode('ascii')])
        self.Process = Sherpa.MEProcess(self.Generator)

        # Incoming flavors must be added first!
        self.Process.AddInFlav(11)
        self.Process.AddInFlav(-11)
        self.Process.AddOutFlav(1)
        self.Process.AddOutFlav(-1)
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

        p1 = xs[:, 0:4]
        p2 = xs[:, 4:8]
        p3 = xs[:, 8:12]

        sample_size = len(xs)
        me = np.empty(sample_size)

        cross_section = np.empty(xs.shape[0])
        for i in range(sample_size):
            momenta = [p1[i], p2[i], p3[i]]
            # apply cuts
            if all([self.pT(p)>self.pT_min for p in momenta]) and all([self.angle(p1, p2)>self.angle_min for p1, p2 in combinations(momenta, 2)]):
                self.Process.SetMomentum(2, p1[i, 0], p1[i, 1], p1[i, 2], p1[i, 3])
                self.Process.SetMomentum(3, p2[i, 0], p2[i, 1], p2[i, 2], p2[i, 3])
                self.Process.SetMomentum(4, p3[i, 0], p3[i, 1], p3[i, 2], p3[i, 3])
                me[i] = self.Process.CSMatrixElement()
            else:
                me[i] = 0.

            cross_section = (2. * np.pi) ** (4.-3. * self.nfinal) / (2. * self.E_CM ** 2) * me

        return self.conversion * cross_section

    # Calculate transverse momentum of 4-vector
    def pT(self, p):
        return np.sqrt(p[1]*p[1]+p[2]*p[2])

    # Calculate angle between two 4-vectors
    def angle(self, p, q):
        cos_angle = (p[1:].dot(q[1:]))/(p[0]*q[0])
        return np.arccos(cos_angle)

# e+ e- -> q qbar g g
class ee_qq_2g(Density):
    def __init__(self, E_CM, pT_min, angle_min):
        ndim = 4 * 4
        self.conversion = 0.389379*1e9  # convert to picobarn
        self.nfinal = 4  # number of final state particles

        super().__init__(ndim, False)

        self.E_CM = E_CM
        self.pT_min = pT_min
        self.angle_min = angle_min

        self.Generator = Sherpa.Sherpa()
        self.Generator.InitializeTheRun(4,
                                        [''.encode('ascii'),
                                         ('RUNDATA=' + CARD2).encode('ascii'),
                                         'INIT_ONLY=2'.encode('ascii'),
                                         'OUTPUT=0'.encode('ascii')])
        self.Process = Sherpa.MEProcess(self.Generator)

        # Incoming flavors must be added first!
        self.Process.AddInFlav(11)
        self.Process.AddInFlav(-11)
        self.Process.AddOutFlav(1)
        self.Process.AddOutFlav(-1)
        self.Process.AddOutFlav(21)
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

        p1 = xs[:, 0:4]
        p2 = xs[:, 4:8]
        p3 = xs[:, 8:12]
        p4 = xs[:, 12:16]

        sample_size = len(xs)
        me = np.empty(sample_size)

        cross_section = np.empty(xs.shape[0])
        for i in range(sample_size):
            momenta = [p1[i], p2[i], p3[i], p4[i]]
            # apply cuts
            if all([self.pT(p)>self.pT_min for p in momenta]) and all([self.angle(p1, p2)>self.angle_min for p1, p2 in combinations(momenta, 2)]):
                self.Process.SetMomentum(2, p1[i, 0], p1[i, 1], p1[i, 2], p1[i, 3])
                self.Process.SetMomentum(3, p2[i, 0], p2[i, 1], p2[i, 2], p2[i, 3])
                self.Process.SetMomentum(4, p3[i, 0], p3[i, 1], p3[i, 2], p3[i, 3])
                self.Process.SetMomentum(5, p4[i, 0], p4[i, 1], p4[i, 2], p4[i, 3])
                me[i] = self.Process.CSMatrixElement()
            else:
                me[i] = 0.

            cross_section = (2. * np.pi) ** (4.-3. * self.nfinal) / (2. * self.E_CM ** 2) * me

        return self.conversion * cross_section

    # Calculate transverse momentum of 4-vector
    def pT(self, p):
        return np.sqrt(p[1]*p[1]+p[2]*p[2])

    # Calculate angle between two 4-vectors
    def angle(self, p, q):
        cos_angle = (p[1:].dot(q[1:]))/(p[0]*q[0])
        return np.arccos(cos_angle)

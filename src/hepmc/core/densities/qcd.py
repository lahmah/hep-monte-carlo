import sys
from Sherpa import Sherpa, MEProcess
from ..density import Density
from ..util import interpret_array
from ..sampling import Sample
import pkg_resources
import numpy as np
from itertools import combinations
import os
import multiprocessing
from multiprocessing.sharedctypes import RawArray

# We need these classes for pickle to work with SWIG. Pickle is 
# used by multiprocessing.
class PickleableSWIG:

    def __setstate__(self, state):
        self.__init__(*state['args'])

    def __getstate__(self):
        return {'args': self.args}

class PickleableSherpa(Sherpa, PickleableSWIG):

    def __init__(self, *args):
        self.args = args
        Sherpa.__init__(self)

class PickleableMEProcess(MEProcess, PickleableSWIG):

    def __init__(self, *args):
        self.args = args
        MEProcess.__init__(self, *args)

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

        self.pin1 = [E_CM/2., 0., 0., E_CM/2.]
        self.pin2 = [E_CM/2., 0., 0., -E_CM/2.]

        self.pool = multiprocessing.Pool(initializer=self.init_worker)

    def __getstate__(self):
        """Remove the pool instance from the __dict__ so that it
        won't be pickled.
        """
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    # The first momentum is xs[0:4]
    # The second momentum is xs[4:8], ...
    def pdf(self, xs):
        xs = interpret_array(xs, self.ndim)

        ndim = xs.shape[1]

        if ndim != self.ndim:
            raise RuntimeWarning("Mismatching dimensions.")

        sample_size = len(xs)

        me = np.zeros(sample_size)
        pass_cut_idx = np.where(cut_pT(xs, self.pT_min) & cut_angle(xs, self.angle_min))
        me[pass_cut_idx] = self.pool.map(self.get_me, xs[pass_cut_idx], chunksize=100)

        cross_section = (2. * np.pi) ** (4.-3. * self.nfinal) / (2. * self.E_CM ** 2) * me

        return self.conversion * cross_section

    def init_worker(self):
        global Process
        Generator = PickleableSherpa()
        Generator.InitializeTheRun(4,
                                    [''.encode('ascii'),
                                     ('RUNDATA=' + path_to_runcard(self.n_gluons)).encode('ascii'),
                                     'INIT_ONLY=2'.encode('ascii'),
                                     'OUTPUT=0'.encode('ascii')])
        Process = PickleableMEProcess(Generator)

        # Incoming flavors must be added first!
        Process.AddInFlav(11)
        Process.AddInFlav(-11)
        Process.AddOutFlav(1)
        Process.AddOutFlav(-1)
        for _ in range(self.n_gluons):
            Process.AddOutFlav(21)
        Process.Initialize()

    def get_me(self, x):
        Process.SetMomenta([self.pin1, self.pin2] + [x[j*4:j*4+4].tolist() for j in range(self.nfinal)])
        return Process.CSMatrixElement()

def path_to_runcard(n_gluons):
    return pkg_resources.resource_filename('hepmc', 'data/ee_qq_' + str(n_gluons) + 'g.dat')

def cut_pT(xs, pT_min):
    n_particles = xs.shape[1] // 4
    pass_cut_flags = np.all([pT(xs[:, i*4:i*4+4]) > pT_min for i in range(n_particles)], axis=0)
    return pass_cut_flags

def cut_angle(xs, angle_min):
    n_particles = xs.shape[1] // 4
    pass_cut_flags = np.all([angle(xs[:, i*4+1:i*4+4], xs[:, j*4+1:j*4+4]) > angle_min for i, j in combinations(range(n_particles), 2)], axis=0)
    return pass_cut_flags

# Calculate transverse momentum of 4-vector
def pT(p):
    return np.sqrt(p[:, 1]*p[:, 1]+p[:, 2]*p[:, 2])

def unit_vector(v):
    return v / np.linalg.norm(v, axis=1)[:, np.newaxis]

# Calculate angle between two 4-vectors
def angle(p, q):
    p_u = unit_vector(p)
    q_u = unit_vector(q)
    cos_angle = np.einsum('ij,ij->i', p_u, q_u)
    return np.arccos(np.clip(cos_angle, -1., 1.))

def export_hepmc(E_CM, sample, filename):
    n_out = int(sample.data.shape[1]/4)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("\nHepMC::Version 2.06.09\nHepMC::IO_GenEvent-START_EVENT_LISTING\n")
        for i in range(sample.size):
            # event
            # E evt_number no_mpi scale alphq_qcd alpha_qed signal_id barcode_signal_process_vertex no_vertices barcode_particle_1 barcode_particle_2 no_random_state {random_state} no_weights {weights}
            f.write("E %i -1 0 1.0000000000000000e+00 1.0000000000000000e+00 0 0 1 10001 10002 0 1 %e\n" % (i, sample.weights[i]))
            
            # weights
            f.write("N 1 \"0\"\n")
            
            # units
            f.write("U GEV MM\n")
            
            # vertex
            # V barcode id x y z ctau no_incoming no_outgoing no_weights {weights}
            f.write("V -1 0 0 0 0 0 2 %i 0\n" % n_out)
            
            # incoming particles
            # P barcode PDG_id px py pz energy gen_mass status_code pol_theta pol_phi barcode_vertex_incoming no_flow {code_index, code}
            f.write("P 10001 11 0 0 %e %e 0 4 0 0 -1 0\n" % (E_CM/2, E_CM/2))
            f.write("P 10002 -11 0 0 %e %e 0 4 0 0 -1 0\n" % (-E_CM/2, E_CM/2))
            
            # outgoing particles
            for j in range(n_out):
                if j == 0:
                    pid = 1
                elif j == 1:
                    pid = -1
                else:
                    pid = 21
                
                E = sample.data[i, 4*j]
                px = sample.data[i, 4*j+1]
                py = sample.data[i, 4*j+2]
                pz = sample.data[i, 4*j+3]
                f.write("P %i %i %e %e %e %e 0 1 0 0 0 0\n" % (10003+j, pid, px, py, pz, E))
            
        f.write("HepMC::IO_GenEvent-END_EVENT_LISTING")

def import_hepmc(filename):
    with open(filename, "r") as f:
        while f.readline() != "HepMC::IO_GenEvent-START_EVENT_LISTING\n":
            pass

        data = []
        weights = []
        line = f.readline()
        while(True):
            weight = float(line.split()[13])
            weights.append(weight)
            line = f.readline()
            while not line.split()[0] == "P":
                line = f.readline()
            pin1 = line
            pin2 = f.readline()

            pout = []
            pids = []
            line = f.readline()
            while line.split()[0] == "P":
                pout.append(line.split()[3:7])
                pids.append(int(line.split()[2]))
                line = f.readline()

            x = []
            # order the particles by id
            p = pout[pids.index(1)]
            x.append(float(p[3]))
            x.append(float(p[0]))
            x.append(float(p[1]))
            x.append(float(p[2]))
            del pout[pids.index(1)]
            del pids[pids.index(1)]
            p = pout[pids.index(-1)]
            x.append(float(p[3]))
            x.append(float(p[0]))
            x.append(float(p[1]))
            x.append(float(p[2]))
            del pout[pids.index(-1)]

            for p in pout:
                x.append(float(p[3]))
                x.append(float(p[0]))
                x.append(float(p[1]))
                x.append(float(p[2]))

            data.append(np.array(x))

            if line == "HepMC::IO_GenEvent-END_EVENT_LISTING\n":
                break

    data = np.array(data)
    weights = np.array(weights)
    return Sample(data=data, weights=weights)

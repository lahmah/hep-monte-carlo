from ..density import Proposal
from hepmc.core.sampling import Sample
from hepmc.core.a_nice_mc.wgan_nll import Trainer
from hepmc.core.a_nice_mc.discriminator import MLPDiscriminator
from hepmc.core.a_nice_mc.generator import create_nice_network
from hepmc.core.a_nice_mc.energy import Energy

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm


class ANiceMC(Proposal):
    def __init__(self, target, training_sample, v_dim=10, train_iters=10000):
        super().__init__(target.ndim, is_symmetric=True)

        energy_fn = EnergyFunction(target)

        discriminator = MLPDiscriminator([400, 400, 400])
        self.generator = create_nice_network(
            target.ndim, v_dim,
            [
                ([400], 'v1', False),
                ([400, 400], 'x1', True),
                ([400], 'v2', False)
            ]
        )

        self.z = tf.placeholder(tf.float32, [1, self.ndim])
        #self.v = tf.placeholder(tf.float32, [1, v_dim])
        self.v = tf.random_normal(shape=[1, self.generator.v_dim])
        self.z_, self.v_ = self.generator([self.z, self.v], is_backward=(tf.random_uniform([]) < 0.5))

        self.trainer = Trainer(training_sample.data, self.generator, energy_fn, discriminator, self.noise_sampler, b=16, m=4)
        self.trainer.train(max_iters=train_iters)

    def noise_sampler(self, bs):
        return np.random.uniform(0., 1., [bs, self.ndim])

    def proposal(self, state):
        #z = tf.placeholder(tf.float32, [1, self.ndim])
        #v = tf.random_normal(shape=[1, self.generator.v_dim])
        #z_, v_ = self.generator([self.z, v], is_backward=(tf.random_uniform([]) < 0.5))
        #z_, v_ = self.trainer.sess.run([z_, v_], feed_dict={self.z: state.reshape((1, self.ndim)), self.v: tf.random_normal(shape=[1, self.generator.v_dim])})
        z_, v_ = self.trainer.sess.run([self.z_, self.v_], feed_dict={self.z: state.reshape((1, self.ndim))})
        #return np.reshape(z_, self.ndim)
        return z_

class EnergyFunction(Energy):
    def __init__(self, target):
        super().__init__()
        self.name = "EnergyFunction"
        self.target = target
        self.z = tf.placeholder(tf.float32, [None, target.ndim], name='z')

    def __call__(self, z):
        return self.tf_energy(z)

    # energy as numpy function
    def energy(self, z):
        return self.target.pot(z).astype(np.float32, copy=False)

    # energy as tensorflow function
    def tf_energy(self, z, name=None):
        with tf.name_scope(name, "energy", [z]) as name:
            y = tf.py_func(self.energy,
                [z],
                [tf.float32],
                name=name,
                stateful=False)
            return y[0]

    def evaluate(self, zv, path=None):
        pass

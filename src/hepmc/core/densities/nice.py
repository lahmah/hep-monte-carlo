from ..density import Distribution
from hepmc.core.sampling import Sample

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
layers = tf.contrib.layers


class Nice(Distribution):
    def __init__(self, training_sample, batch_size=100, num_bijectors=4, train_iters=1e4, hidden_layers=[512, 512]):
        super().__init__(training_sample.data.shape[1])

        if self.ndim % 2 == 0 and num_bijectors % 2 != 0:
            raise ValueError("If the number of dimensions is even, the number of bijectors must be a multiple of 2.")
        if self.ndim % 2 != 0 and num_bijectors % 5 != 0:
            raise ValueError("If the number of dimensions is odd, the number of bijectors must be a multiple of 5.")

        DTYPE = tf.float32
        self.NP_DTYPE = np.float32

        X = training_sample.data
        dataset = tf.data.Dataset.from_tensor_slices(X.astype(self.NP_DTYPE))
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=X.shape[0])
        dataset = dataset.prefetch(3 * batch_size)
        dataset = dataset.batch(batch_size)
        data_iterator = dataset.make_one_shot_iterator()
        x_samples = data_iterator.get_next()

        self.base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([self.ndim], DTYPE))
        bijectors = []
        for i in range(num_bijectors):
            bijectors.append(tfb.RealNVP(
                num_masked = -(-self.ndim // 2), # ceil division
                shift_and_log_scale_fn=tfb.real_nvp_default_template(hidden_layers=hidden_layers)))
            # swap the right and left halves of the dimensions
            bijectors.append(tfb.Permute(
                permutation=list(range(self.ndim//2, self.ndim))+list(range(0, self.ndim//2))))
        # Discard the last Permute layer
        flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))

        self.dist = tfd.TransformedDistribution(
                distribution=self.base_dist,
                bijector=flow_bijector)

        loss = -tf.reduce_mean(self.dist.log_prob(x_samples))
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        NUM_STEPS = int(train_iters)
        global_step = []
        np_losses = []
        for i in range(NUM_STEPS):
            _, np_loss = self.sess.run([train_op, loss])
            if i % 1000 == 0:
                global_step.append(i)
                np_losses.append(np_loss)
                print(i, np_loss)
        plt.plot(np_losses) # plot the training progress

    def pdf(self, xs):
        pdf = self.dist.prob(xs.astype(self.NP_DTYPE))
        return self.sess.run(pdf)

    def rvs(self, sample_count):
        x = self.base_dist.sample(sample_count)
        for bijector in reversed(self.dist.bijector.bijectors):
            x = bijector.forward(x)
        result = self.sess.run(x)
        return result

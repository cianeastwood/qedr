import os, sys
import time
import re
import numpy as np
import tensorflow as tf
sys.path.append("..")
sys.path.append("../..")

from lib.models import params_with_name
from lib.models.save_images import save_images
from lib.models.distributions import Bernoulli, Gaussian, Product
from lib.models.nets_64x64 import NetsRetreiver

TINY = 1e-8
SEED = 123

class VAE(object):
    def __init__(self, session, output_dist, z_dist, arch, batch_size, image_shape, exp_name, dirs,
                 gaps, beta, vis_reconst, vis_disent, n_disentangle_samples):
        """
        :type output_dist: Distribution
        :type z_dist: Gaussian
        """
        self.session = session
        self.output_dist = output_dist
        self.z_dist = z_dist
        self.arch = arch
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.exp_name = exp_name
        self.dirs = dirs
        self.beta = beta
        self.gaps = gaps
        self.vis_reconst = vis_reconst
        self.vis_disent = vis_disent
        self.n_disentangle_samples = n_disentangle_samples

        self.__build_graph()

    def __build_graph(self):
        tf.set_random_seed(SEED)
        np.random.seed(SEED)
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.int32, shape=[None] + list(self.image_shape))

        # Normalize + reshape 'real' input data
        norm_x = 2*((tf.cast(self.x, tf.float32)/255.)-.5)

        # Set Encoder and Decoder archs
        self.Encoder, self.Decoder = NetsRetreiver(self.arch)

        # Encode
        self.z_dist_info, self.z = self.__Enc(norm_x)

        # Decode
        self.x_out, x_out_logit = self.__Dec(self.z)

        # Loss and optimizer
        self.__prep_loss_optimizer(norm_x, x_out_logit)

    def __Enc(self, x):
        z_dist_params = self.Encoder('Encoder', x, self.image_shape[0], self.z_dist.dist_flat_dim,
                                          self.is_training)
        z_dist_info = self.z_dist.activate_dist(z_dist_params)
        if isinstance(self.z_dist, Gaussian):
            z = self.z_dist.sample(z_dist_info)
        else:
            raise NotImplementedError #reparam trick for other latent dists
        return z_dist_info, z

    def __Dec(self, z):
        x_out_logit = self.Decoder('Decoder', z, self.image_shape[0], self.is_training)
        if isinstance(self.output_dist, Gaussian):
            x_out = tf.tanh(x_out_logit)
        elif isinstance(self.output_dist, Bernoulli):
            x_out = tf.nn.sigmoid(x_out_logit)
        else:
            raise Exception()
        return x_out, x_out_logit

    def __prep_loss_optimizer(self, norm_x, x_out_logit):
        norm_x = tf.reshape(norm_x, [-1, self.output_dist.dim])

        # reconstruction loss
        if isinstance(self.output_dist, Gaussian):
            reconstr_loss =  tf.reduce_sum(tf.square(norm_x - self.x_out), axis=1)
        elif isinstance(self.output_dist, Bernoulli):
            reconstr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x,
                                                                      logits=x_out_logit)
            reconstr_loss = tf.reduce_sum(reconstr_loss, 1)
        else:
            raise Exception()

        # latent loss
        kl_post_prior = self.z_dist.kl(self.z_dist_info, self.z_dist.prior_dist_info(tf.shape(self.z)[0]))

        # average over batch
        self.loss = tf.reduce_mean(reconstr_loss + self.beta * kl_post_prior)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(self.loss)

    def load(self):
        self.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.dirs['ckpt'])

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = ckpt.model_checkpoint_path
            self.saver.restore(self.session, ckpt_name)
            print("Checkpoint restored: {0}".format(ckpt_name))
            prev_step = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))

        else:
            print("Failed to find checkpoint.")
            prev_step = 0
        sys.stdout.flush()
        return prev_step + 1

    def train(self, n_iters, n_iters_per_epoch, stats_iters, ckpt_interval):
        self.session.run(tf.global_variables_initializer())

        # Fixed GT samples - save
        fixed_x, _ = next(self.train_iter)
        fixed_x = self.session.run(tf.constant(fixed_x))
        save_images(fixed_x, os.path.join(self.dirs['samples'], 'samples_groundtruth.png'))

        start_iter = self.load()
        running_cost = 0.

        for iteration in range(start_iter, n_iters):
            start_time = time.time()

            _data, _ = next(self.train_iter)
            _, cost = self.session.run((self.optimizer, self.loss), feed_dict={self.x: _data, self.is_training:True})
            running_cost += cost

            if iteration % n_iters_per_epoch == 1:
                print("Epoch: {0}".format(iteration // n_iters_per_epoch))

            # Print avg stats and dev set stats
            if (iteration < start_iter + 4) or iteration % stats_iters == 0:
                t = time.time()
                dev_data, _ = next(self.dev_iter)
                dev_cost, dev_z_dist_info = self.session.run([self.loss, self.z_dist_info],
                                                     feed_dict={self.x: dev_data, self.is_training:False})

                n_samples = 1. if (iteration < start_iter + 4) else float(stats_iters)
                avg_cost = running_cost / n_samples
                running_cost = 0.
                print("Iteration:{0} \t| Train cost:{1:.1f} \t| Dev cost: {2:.1f}".format(iteration, avg_cost, dev_cost))

                if isinstance(self.z_dist, Gaussian):
                    avg_dev_var = np.mean(dev_z_dist_info["stddev"]**2, axis=0)
                    zss_str = ""
                    for i,zss in enumerate(avg_dev_var):
                       z_str = "z{0}={1:.2f}".format(i,zss)
                       zss_str += z_str + ", "
                    print("z variance:{0}".format(zss_str))

                if self.vis_reconst:
                    self.visualise_reconstruction(fixed_x)
                if self.vis_disent:
                    self.visualise_disentanglement(fixed_x[0])

                if np.any(np.isnan(avg_cost)):
                    raise ValueError("NaN detected!")

            if (iteration > start_iter) and iteration % (ckpt_interval) == 0:
                self.saver.save(self.session, os.path.join(self.dirs['ckpt'], self.exp_name), global_step=iteration)

    def encode(self, X, is_training=False):
        """Encode data, i.e. map it into latent space."""
        [z_dist_info] = self.session.run([self.z_dist_info],
                                         feed_dict={self.x: X, self.is_training: is_training})
        if isinstance(self.z_dist, Gaussian):
            code = z_dist_info["mean"]
        else:
            raise NotImplementedError
        return code

    def reconstruct(self, X, is_training=False):
        """ Reconstruct data. """
        return self.session.run(self.x_out,
                                feed_dict={self.x: X, self.is_training: is_training})

    def generate(self, z_mu=None, batch_size=None, is_training=False):
        """ Generate data from code or latent representation."""
        if z_mu is None:
            batch_size = self.batch_size if batch_size is None else batch_size
            z_mu = self.arch.reg_latent_dist.sample_prior(batch_size)
        return self.session.run(self.x_out, feed_dict={self.z: z_mu, self.is_training: is_training})

    def visualise_reconstruction(self, X):
        X_r = self.reconstruct(X)
        X_r = ((X_r+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(X_r, os.path.join(self.dirs['samples'], 'samples_reconstructed.png'))

    def visualise_disentanglement(self, x):
        z = self.encode([x])
        n_zs = z.shape[1] #self.latent_dist.dim
        z = z[0]
        rimgs = []

        if isinstance(self.z_dist, Gaussian):
            for target_z_index in range(n_zs):
                for ri in range(self.n_disentangle_samples):
                    value = -3.0 + 6.0 / (self.n_disentangle_samples-1.) * ri
                    z_new = np.zeros((1, n_zs))
                    for i in range(n_zs):
                        if (i == target_z_index):
                            z_new[0][i] = value
                        else:
                            z_new[0][i] = z[i]
                    rimgs.append(self.generate(z_mu=z_new))
        else:
            raise NotImplementedError
        rimgs = np.vstack(rimgs).reshape([n_zs, self.n_disentangle_samples, -1]) #.transpose(1,0,2)
        rimgs = rimgs[[5,7,2,6,1,9,3]] #order of zs captured
        rimgs = ((rimgs+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(rimgs, os.path.join(self.dirs['samples'], 'disentanglement.png'),
                    n_cols=n_zs, n_rows=self.n_disentangle_samples)

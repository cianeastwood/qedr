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

class RegularisedGAN(object):
    def __init__(self, session, output_dist, z_dist, c_dist, arch, batch_size,
                 image_shape, exp_name, dirs, mi_coeff, gp_coeff, mode,
                 critic_iters, gaps, vis_reconst, vis_disent, n_disentangle_samples):
        """
        :type output_dist: Distribution
        :type z_dist: Distribution
        :type c_dist: Gaussian
        """
        self.session = session
        self.output_dist = output_dist
        self.z_dist = z_dist
        self.c_dist = c_dist
        self.arch = arch
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.exp_name = exp_name
        self.dirs = dirs
        self.mi_coeff = mi_coeff
        self.gp_coeff = gp_coeff
        self.mode = mode
        self.critic_iters = critic_iters
        self.gaps = gaps
        self.vis_reconst = vis_reconst
        self.vis_disent = vis_disent
        self.n_disentangle_samples = n_disentangle_samples

        self.latent_dist = Product([z_dist, c_dist])
        self.log_vars = []
        self.__build_graph()

    def __build_graph(self):
        tf.set_random_seed(SEED)
        np.random.seed(SEED)
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.int32, shape=[None] + list(self.image_shape))

        # Normalize + reshape 'real' input data
        real_x = 2*((tf.cast(self.x, tf.float32)/255.)-.5)
        real_x = tf.reshape(real_x, [-1, self.output_dist.dim])

        # Sample prior (z,c)
        self.z = tf.placeholder_with_default(self.z_dist.sample_prior(self.batch_size), [None, self.z_dist.dim])
        self.c = tf.placeholder_with_default(self.c_dist.sample_prior(self.batch_size), [None, self.c_dist.dim])
        self.z_c = tf.concat([self.z,self.c], axis=1)

        # Set Gen and Disc/Q arch
        self.Discriminator_Q, self.Generator = NetsRetreiver(self.arch)

        # Generate 'fake' data
        self.fake_x = self.__G(self.z_c)

        # Discriminate real data and 'encode' to posterior Q(c|x)
        disc_real, self.q_c_given_x_real_dist_info = self.__D_Q(real_x)

        # Discriminate fake data and 'encode' to posterior Q(c|x)
        disc_fake, q_c_given_x_fake_dist_info = self.__D_Q(self.fake_x)

        # Loss and optimizer
        self.__prep_loss_optimizer(real_x, disc_real, disc_fake, q_c_given_x_fake_dist_info)

    def __G(self, z_c):
        x_out_logit = self.Generator('Generator', z_c, self.image_shape[0], self.is_training, self.mode)
        if isinstance(self.output_dist, Gaussian):
            x_fake = tf.tanh(x_out_logit)
        elif isinstance(self.output_dist, Bernoulli):
            x_fake = tf.nn.sigmoid(x_out_logit)
        else:
            raise Exception()
        return x_fake

    def __D_Q(self, x):
        output_D_Q = self.Discriminator_Q('Discriminator', x, self.image_shape[0],
                                          self.c_dist.dist_flat_dim + 1, self.is_training, self.mode)
        d_out = output_D_Q[:, :1] # D_output, dim=1
        q_out = output_D_Q[:, 1:] # Q_output, params of Q(c|x)
        return d_out, self.c_dist.activate_dist(q_out)

    def __prep_loss_optimizer(self, real_x, disc_real, disc_fake, q_c_given_x_dist_info):
        # Gen / disc costs
        if self.mode == 'wgan-gp':
            self.gen_cost = -tf.reduce_mean(disc_fake)
            self.disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
            alpha = tf.random_uniform(
                shape=[self.batch_size,1],
                minval=0.,
                maxval=1.,
                seed = SEED,
            )
            differences = self.fake_x - real_x
            interpolates = real_x + (alpha*differences)
            d_hat, _ = self.__D_Q(interpolates)
            gradients = tf.gradients(d_hat, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            self.disc_cost += self.gp_coeff * gradient_penalty

        else: #vanilla / DC GAN
            self.gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                              labels=tf.ones_like(disc_fake)))
            self.disc_cost =  0.5 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                labels=tf.zeros_like(disc_fake)))
            self.disc_cost += 0.5 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                                labels=tf.ones_like(disc_real)))
        # Mutual information
        log_q_c_given_x = self.c_dist.logli(self.c, q_c_given_x_dist_info)
        log_q_c = self.c_dist.logli_prior(self.c)
        cross_entropy = tf.reduce_mean(-log_q_c_given_x)
        entropy = tf.reduce_mean(-log_q_c)
        self.mi_est = entropy - cross_entropy
        self.disc_cost -= self.mi_coeff * self.mi_est
        self.gen_cost -= self.mi_coeff * self.mi_est

        # Log vars
        self.log_vars.append(("Discriminator loss", self.disc_cost))
        self.log_vars.append(("Generator loss", self.gen_cost))
        self.log_vars.append(("MI", self.mi_est))

        # Optimizers
        if self.mode == 'wgan-gp':
            self.gen_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(self.gen_cost,
                                              var_list=params_with_name('Generator'))
            self.disc_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(self.disc_cost,
                                               var_list=params_with_name('Discriminator.'))

        else: #vanilla / DC GAN
            self.gen_opt = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.gen_cost,
                                              var_list=params_with_name('Generator'))
            self.disc_opt = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.disc_cost,
                                               var_list=params_with_name('Discriminator.'))

    def __get_code(self, dist_info):
        if isinstance(self.c_dist, Gaussian):
            return dist_info["mean"]
        else:
            raise NotImplementedError

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

    def encode(self, X, is_training=False):
        """Encode data, i.e. map it into latent space."""
        [c_dist_info] = self.session.run([self.q_c_given_x_real_dist_info],
                                       feed_dict={self.x: X, self.is_training: is_training})
        return self.__get_code(c_dist_info)

    def reconstruct(self, X, z=None, is_training=False):
        """Reconstruct data."""
        code = self.encode(X)
        if z is None:
            return self.generate(c=code)
        z_c = np.concatenate([z, code], axis=1)
        return self.generate(z_c=z_c)

    def generate(self, z_c=None, c=None, batch_size=None, is_training=False):
        """Generate data from latent representation (z,c) or c only."""
        if c is None:
            if z_c is None:
                batch_size = self.batch_size if batch_size is None else batch_size
                z_c = self.session.run(self.latent_dist.sample_prior(batch_size))
        else:
            z = self.session.run(self.z_dist.sample_prior(c.shape[0]))
            z_c = np.concatenate([z,c], axis=1)
        return self.session.run(self.fake_x,
                                feed_dict={self.z_c: z_c, self.is_training: is_training})

    def train(self, n_iters, n_iters_per_epoch, stats_iters, ckpt_interval):
        self.session.run(tf.global_variables_initializer())

        # Fixed GT samples
        fixed_x, _ = next(self.train_iter)
        fixed_x = self.session.run(tf.constant(fixed_x))
        save_images(fixed_x, os.path.join(self.dirs['samples'], 'samples_groundtruth.png'))

        # Fixed prior sample, for generating samples
        fixed_z_c = self.session.run(self.latent_dist.sample_prior(self.batch_size))

        log_vars = [x for _, x in self.log_vars]
        log_keys = [x for x, _ in self.log_vars]
        running_log_vals = [0.] * len(self.log_vars)

        start_iter = self.load()

        for iteration in range(start_iter, n_iters):
            start_time = time.time()

            # Train generator
            if iteration > start_iter:
                _ = self.session.run(self.gen_opt, feed_dict={self.x: _data, self.is_training:True})

            # Train critic
            if self.mode == 'wgan-gp':
                disc_iters = self.critic_iters

            else:
                disc_iters = 1

            for i in range(disc_iters):
                _data, _ = next(self.train_iter)
                log_vals = self.session.run([self.disc_opt] + log_vars,
                                            feed_dict={self.x: _data, self.is_training:True})[1:]

            running_log_vals = [running + batch for running, batch in zip(running_log_vals, log_vals)]

            if iteration % n_iters_per_epoch == 1:
                print("Epoch: {0}".format(iteration // n_iters_per_epoch))

            # Print avg stats and dev set stats
            if (iteration < start_iter + 4) or iteration % stats_iters == 0:
                t = time.time()
                dev_data, _ = next(self.dev_iter)
                dev_log_vals = self.session.run(log_vars + [self.q_c_given_x_real_dist_info],
                                                 feed_dict={self.x: dev_data, self.is_training:False})
                dev_c_dist_info = dev_log_vals[-1]
                dev_log_vals = dev_log_vals[:-1]

                n_samples = 1. if (iteration < start_iter + 4) else float(stats_iters)
                avg_log_vals = [r_log_val / n_samples for r_log_val in running_log_vals]
                running_log_vals = [0.] * len(self.log_vars)

                tr_log_line = " | ".join("{0}: {1:.2f}".format(k, v) for k, v in zip(log_keys, avg_log_vals))
                dev_log_line = " | ".join("{0}: {1:.2f}".format(k, v) for k, v in zip(log_keys, dev_log_vals))
                print("Iteration:{0} \nTrain: {1} \nDev  : {2}".format(iteration, tr_log_line, dev_log_line))

                if isinstance(self.c_dist, Gaussian) and not self.c_dist._fix_std:
                    avg_dev_var = np.mean(dev_c_dist_info["stddev"]**2, axis=0)
                    cs_str = ""
                    for i, cs in enumerate(avg_dev_var):
                        c_str = "c{0}={1:.2f}".format(i,cs)
                        cs_str += c_str + ", "
                    print("c variance:{0}".format(cs_str))

                fixed_samples = self.generate(z_c=fixed_z_c).reshape([-1] + self.image_shape)
                save_images(fixed_samples, os.path.join(self.dirs['samples'], 'samples_generated.png'))

                if self.vis_reconst:
                    self.visualise_reconstruction(fixed_x, fixed_z_c)
                if self.vis_disent:
                    self.visualise_disentanglement(fixed_x[0], fixed_z_c)

                if np.any(np.isnan(avg_log_vals)):
                    raise ValueError("NaN detected!")

            if (iteration > start_iter) and iteration % (ckpt_interval) == 0:
                self.saver.save(self.session, os.path.join(self.dirs['ckpt'], self.exp_name), global_step=iteration)

    def visualise_reconstruction(self, X, fixed_z_c):
        fixed_z, fixed_c = self.latent_dist.split_var(fixed_z_c)
        X_r = self.reconstruct(X, z=fixed_z)
        X_r = ((X_r+1.)*(255.99/2)).astype('int32')
        save_images(X_r.reshape([-1] + self.image_shape),
                    os.path.join(self.dirs['samples'], 'samples_reconstructed.png'))

    def visualise_disentanglement(self, x, fixed_z_c):
        c = self.encode([x])
        n_cs = c.shape[1] # self.c_dist.dim
        c = c[0]
        fixed_z, fixed_c = self.latent_dist.split_var(fixed_z_c[:1])
        rimgs = []

        if isinstance(self.c_dist, Gaussian):
            for target_c_index in range(n_cs):
                for ri in range(self.n_disentangle_samples):
                    value = -1.0 + 2.0 / (self.n_disentangle_samples-1.) * ri
                    c_new = np.zeros((1, n_cs))
                    for i in range(n_cs):
                        if (i == target_c_index):
                            c_new[0][i] = value
                        else:
                            c_new[0][i] = c[i]
                    z_c_new = np.concatenate([fixed_z, c_new], axis=1)
                    rimgs.append(self.generate(z_c=z_c_new))
        else:
            raise NotImplementedError

        rimgs = np.vstack(rimgs).reshape([n_cs, self.n_disentangle_samples, -1])
        rimgs = ((rimgs+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(rimgs, os.path.join(self.dirs['samples'], 'disentanglement.png'),
                    n_rows=n_cs, n_cols=self.n_disentangle_samples)

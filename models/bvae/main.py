import os, sys
import numpy as np
import tensorflow as tf
from operator import mul
from functools import reduce

from model import VAE
sys.path.append("..")
sys.path.append("../..")
from lib.models.distributions import Gaussian
from lib.utils import init_directories, create_directories
from lib.models.data_managers import TeapotsDataManager

flags = tf.app.flags
flags.DEFINE_integer("epochs", 50, "Number of epochs to train [25]")
flags.DEFINE_integer("stats_interval", 1., "Print/log stats every [stats_interval] epochs. [1.0]")
flags.DEFINE_integer("ckpt_interval", 10, "Save checkpoint every [ckpt_interval] epochs. [10]")
flags.DEFINE_integer("latent_dim", 10, "Number of latent variables [10]")
flags.DEFINE_float("beta", 1., "D_KL term weighting [1.]")
flags.DEFINE_integer("batch_size", 64, "The size of training batches [64]")
flags.DEFINE_string("image_shape", "(3,64,64)", "Shape of inputs images [(3,64,64)]")
flags.DEFINE_string("exp_name", None, "The name of experiment [None]")
flags.DEFINE_string("arch", "resnet", "The desired arch: low_cap, high_cap, resnet. [resnet]")
flags.DEFINE_string("output_dir", "./", "Output directory for checkpoints, samples, etc. [.]")
flags.DEFINE_string("data_dir", None, "Data directory [None]")
flags.DEFINE_string("file_ext", ".jpeg", "Image filename extension [.jpeg]")
flags.DEFINE_boolean("gaps", True, "Create gaps in data to faciliate zero-shot inference [False]")
flags.DEFINE_boolean("train", True, "Train [True]")
flags.DEFINE_boolean("save_codes", False, "Save latent representation or code for all data samples [False]")
flags.DEFINE_boolean("visualize_reconstruct", True, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("visualize_disentangle", True, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("n_disentangle_samples", 10, "The number of evenly spaced samples in latent space \
                     over the interval [-3, 3] [64]")
FLAGS = flags.FLAGS

def main(_):
    if FLAGS.exp_name is None:
        FLAGS.exp_name = "vae_{0}_{1}_{2}_{3}".format(FLAGS.gaps, FLAGS.arch, FLAGS.latent_dim, FLAGS.beta)
    image_shape = [int(i) for i in FLAGS.image_shape.strip('()[]{}').split(',')]
    dirs = init_directories(FLAGS.exp_name, FLAGS.output_dir)
    dirs['data'] = '../../data' if FLAGS.data_dir is None else FLAGS.data_dir
    dirs['codes'] = os.path.join(dirs['data'], 'codes/')
    create_directories(dirs, FLAGS.train, FLAGS.save_codes)

    z_dist = Gaussian(FLAGS.latent_dim)
    output_dim  = reduce(mul, image_shape, 1)
    output_dist = Gaussian(output_dim)

    run_config = tf.ConfigProto(allow_soft_placement=True)
    run_config.gpu_options.allow_growth=True
    run_config.gpu_options.per_process_gpu_memory_fraction=0.9
    sess = tf.Session(config=run_config)

    vae = VAE(
        session=sess,
        output_dist=output_dist,
        z_dist=z_dist,
        arch=FLAGS.arch,
        batch_size=FLAGS.batch_size,
        image_shape=image_shape,
        exp_name=FLAGS.exp_name,
        dirs=dirs,
        beta=FLAGS.beta,
        gaps=FLAGS.gaps,
        vis_reconst=FLAGS.visualize_reconstruct,
        vis_disent=FLAGS.visualize_disentangle,
        n_disentangle_samples=FLAGS.n_disentangle_samples,
    )

    if FLAGS.train:
        data_manager = TeapotsDataManager(dirs['data'], FLAGS.batch_size,
                              image_shape, shuffle=True, gaps=FLAGS.gaps,
                              file_ext=FLAGS.file_ext, train_fract=0.8,
                              inf=True)
        vae.train_iter, vae.dev_iter, vae.test_iter = data_manager.get_iterators()

        n_iters_per_epoch = data_manager.n_train // data_manager.batch_size
        FLAGS.stats_interval = int(FLAGS.stats_interval * n_iters_per_epoch)
        FLAGS.ckpt_interval = int(FLAGS.ckpt_interval * n_iters_per_epoch)
        n_iters = int(FLAGS.epochs * n_iters_per_epoch)

        vae.train(n_iters, n_iters_per_epoch, FLAGS.stats_interval, FLAGS.ckpt_interval)

    if FLAGS.save_codes:
        b_size = 500 #large batch, forward prop only
        data_manager = TeapotsDataManager(dirs['data'], b_size, image_shape, shuffle=False, gaps=False,
                                          file_ext=FLAGS.file_ext, train_fract=1., inf=False)
        data_manager.set_divisor_batch_size()
        vae.train_iter, vae.dev_iter, vae.test_iter = data_manager.get_iterators()

        vae.session.run(tf.global_variables_initializer())
        saved_step = vae.load()
        assert saved_step > 1, "A trained model is needed to encode the data!"

        codes = []
        for batch_num, (img_batch, _) in enumerate(vae.train_iter):
            code = vae.encode(img_batch) #[batch_size, reg_latent_dim]
            codes.append(code)
            if batch_num < 5 or batch_num % 100 == 0:
                print(("Batch number {0}".format(batch_num)))

        codes = np.vstack(codes)
        filename = os.path.join(dirs['codes'], "codes_" + FLAGS.exp_name)
        np.save(filename, codes)
        print(("Codes saved to: {0}".format(filename)))

if __name__ == '__main__':
    tf.app.run()

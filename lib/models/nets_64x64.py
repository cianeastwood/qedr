'''Based on https://github.com/igul222/improved_wgan_training/blob/master/gan_64x64.py'''

import functools
import tensorflow as tf

from lib.models.distributions import Product, Distribution, Gaussian, Categorical, Bernoulli, MeanBernoulli
from lib.models.ops.linear import Linear
from lib.models.ops.conv2d import Conv2D
from lib.models.ops.deconv2d import Deconv2D
from lib.models.ops import linear, conv2d, deconv2d
from lib.models.ops.batchnorm import Batchnorm
from lib.models.ops.layernorm import Layernorm

DIM = 64
CPU = False # True: data_format='NHWC', False: data_format='NCHW'

def LeakyReLU(x, alpha=0.2):
        return tf.maximum(alpha*x, x)

def Normalize(name, axes, inputs, is_training, mode):
    if ('Discriminator' in name) and (mode == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return Layernorm(name,[1,2,3],inputs)
    else:
        return Batchnorm(name,axes,inputs,fused=True, cpu=CPU, is_training=is_training)

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases, cpu=CPU)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases, cpu=CPU)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases, cpu=CPU)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, is_training, mode, 
                  resample=None, he_init=True, norm_inputs=False):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(Conv2D, input_dim=input_dim, output_dim=input_dim, cpu=CPU)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(Conv2D, input_dim=output_dim, output_dim=output_dim, cpu=CPU)
    elif resample==None:
        conv_shortcut = Conv2D
        conv_1        = functools.partial(Conv2D, input_dim=input_dim,  output_dim=input_dim, cpu=CPU)
        conv_2        = functools.partial(Conv2D, input_dim=input_dim, output_dim=output_dim, cpu=CPU)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    if not norm_inputs:
        output = Normalize(name+'.BN1', [0,2,3], output, is_training, mode)
        output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name+'.BN2', [0,2,3], output, is_training, mode)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output


# --- Low capacity net: Arch from 'B-VAE' paper: https://openreview.net/pdf?id=Sy2fzU9gl ---
def low_capacity_encoder(name, inputs, n_channels, latent_dim, is_training, mode=None, nonlinearity=tf.nn.relu):
    output = tf.reshape(inputs, [-1, n_channels, DIM, DIM])
    output = Conv2D(name + '.0', n_channels, DIM // 2, 4, output, stride=2)
    output = Normalize(name + '.BN0', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Conv2D(name + '.1',  DIM // 2, DIM // 2, 4, output, stride=2)
    output = Normalize(name + '.BN1', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Conv2D(name + '.2', DIM // 2, DIM , 4, output, stride=2)
    output = Normalize(name + '.BN2', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Conv2D(name + '.3', DIM, DIM, 4, output, stride=2)
    output = Normalize(name + '.BN3', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = tf.reshape(output, [-1, DIM*4*4])
    output = Linear(name + '.FC', DIM*4*4, DIM*4, output)
    output = Normalize(name + '.BNFC', [0], output, is_training, mode)
    output = nonlinearity(output)

    output = Linear(name + '.Output', DIM*4, latent_dim, output)
    return output

def low_capacity_decoder(name, z, n_channels, is_training, mode=None, nonlinearity=tf.nn.relu):
    output = Linear(name + '.Input', z.get_shape().as_list()[1], DIM*4, z)
    output = Normalize(name + '.BN0', [0], output, is_training, mode)
    output = nonlinearity(output)
    output = tf.reshape(output, [-1, DIM // 4, 4, 4])

    output = Deconv2D(name + '.1', DIM // 4, DIM, 4, output)
    output = Normalize(name + '.BN1', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Deconv2D(name + '.2', DIM, DIM, 4, output)
    output = Normalize(name + '.BN2', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Deconv2D(name + '.3', DIM, DIM // 2, 4, output)
    output = Normalize(name + '.BN3', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Deconv2D(name + '.4', DIM // 2, n_channels, 4, output)
    output = tf.reshape(output, [-1, n_channels*DIM*DIM])
    return output


# --- High capacity net: Arch from 'autoencoding beyond pixels' paper: https://arxiv.org/pdf/1512.09300.pdf ---
def high_capacity_encoder(name, inputs, n_channels, latent_dim, is_training, mode=None, nonlinearity=tf.nn.relu):
    output = tf.reshape(inputs, [-1, n_channels, DIM, DIM])
    output = Conv2D(name + '.0', n_channels, DIM, 5, output, stride=2)
    output = Normalize(name + '.BN0', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Conv2D(name + '.1',  DIM, DIM*2, 5, output, stride=2)
    output = Normalize(name + '.BN1', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Conv2D(name + '.2', DIM*2, DIM*4, 5, output, stride=2)
    output = Normalize(name + '.BN2', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = tf.reshape(output, [-1, DIM*4*8*8])
    output = Linear(name + '.FC', DIM*4*8*8, DIM*4*8, output)
    output = Normalize(name + '.BNFC', [0], output, is_training, mode)
    output = nonlinearity(output)

    output = Linear(name + '.Output', DIM*4*8, latent_dim, output)
    return output     

def high_capacity_decoder(name, z, n_channels, is_training, mode=None, nonlinearity=tf.nn.relu):
    output = Linear(name + '.Input', z.get_shape().as_list()[1], DIM*4*8*8, z)
    output = Normalize(name + '.BN0', [0], output, is_training, mode)
    output = nonlinearity(output)
    output = tf.reshape(output, [-1, DIM*4, 8, 8])

    output = Deconv2D(name + '.1', DIM*4, DIM*4, 5, output)
    output = Normalize(name + '.BN1', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Deconv2D(name + '.2', DIM*4, DIM*2, 5, output)
    output = Normalize(name + '.BN2', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Deconv2D(name + '.3', DIM*2, DIM // 2, 5, output)
    output = Normalize(name + '.BN3', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Conv2D(name + '.4', DIM // 2, n_channels, 5, output)
    output = tf.reshape(output, [-1, n_channels*DIM*DIM])
    return output


# --- ResNet: Arch from improved WGAN paper: https://arxiv.org/pdf/1704.00028.pdf ---     
def resnet_encoder(name, inputs, n_channels, latent_dim, is_training, mode=None, nonlinearity=tf.nn.relu):
    output = tf.reshape(inputs, [-1, n_channels, DIM, DIM])
    output = Conv2D(name + '.Input', n_channels, DIM, 3, output, he_init=False, cpu=CPU)

    output = ResidualBlock(name + '.Res1', DIM, 2*DIM, 3, output, is_training, mode, resample='down')
    output = ResidualBlock(name + '.Res2', 2*DIM, 4*DIM, 3, output, is_training, mode, resample='down')
    output = ResidualBlock(name + '.Res3', 4*DIM, 8*DIM, 3, output, is_training, mode, resample='down')
    output = ResidualBlock(name + '.Res4', 8*DIM, 8*DIM, 3, output, is_training, mode, resample='down')    
    output = tf.reshape(output, [-1, 4*4*8*DIM])
    output = Linear(name + '.Output', 4*4*8*DIM, latent_dim, output)
    
    return output

def resnet_encoder_new(name, inputs, n_channels, latent_dim, is_training, mode=None, nonlinearity=tf.nn.relu):
    output = tf.reshape(inputs, [-1, n_channels, DIM, DIM])
    output = Conv2D(name + '.Input', n_channels, DIM, 3, output, he_init=False, cpu=CPU)

    output = ResidualBlock(name + '.Res1', DIM, 2*DIM, 3, output, is_training, mode, resample='down')
    output = ResidualBlock(name + '.Res2', 2*DIM, 4*DIM, 3, output, is_training, mode, resample='down')
    output = ResidualBlock(name + '.Res3', 4*DIM, 8*DIM, 3, output, is_training, mode, resample='down')
    output = ResidualBlock(name + '.Res4', 8*DIM, 8*DIM, 3, output, is_training, mode, resample='down')
    output = Normalize(name + '.BN.5', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)
    output = tf.reshape(output, [-1, 4*4*8*DIM])
    output = Linear(name + '.Output', 4*4*8*DIM, latent_dim, output)
    
    return output

def resnet_decoder(name, z, n_channels, is_training, mode=None, nonlinearity=tf.nn.relu):
    output = Linear(name + '.Input', z.get_shape().as_list()[1], 4*4*8*DIM, z)
    output = Normalize(name + '.BN0', [0], output, is_training, mode)
    output = nonlinearity(output)
    output = tf.reshape(output, [-1, 8*DIM, 4, 4])

    output = ResidualBlock(name + '.Res1', 8*DIM, 8*DIM, 3, output, is_training, mode, resample='up', norm_inputs=True)
    output = ResidualBlock(name + '.Res2', 8*DIM, 4*DIM, 3, output, is_training, mode, resample='up')
    output = ResidualBlock(name + '.Res3', 4*DIM, 2*DIM, 3, output, is_training, mode, resample='up')
    output = ResidualBlock(name + '.Res4', 2*DIM, 1*DIM, 3, output, is_training, mode, resample='up')

    output = Normalize(name + '.BN5', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)
    output = Conv2D(name + '.Output', DIM, n_channels, 3, output)
    output = tf.reshape(output, [-1, n_channels*DIM*DIM])        
    return output   

# --- DCGAN: Arch from DCGAN paper: https://arxiv.org/pdf/1511.06434.pdf  ---
# i.e. Discriminator + Q(c|x)
def dcgan_encoder(name, inputs, n_channels, latent_dim, is_training, mode=None, nonlinearity=LeakyReLU):
    conv2d.set_weights_stdev(0.02)
    deconv2d.set_weights_stdev(0.02)
    linear.set_weights_stdev(0.02)
    
    output = tf.reshape(inputs, [-1, n_channels, DIM, DIM])
    output = Conv2D(name + '.1', 3, DIM, 5, output, stride=2)
    output = nonlinearity(output)

    output = Conv2D(name + '.2', DIM, 2*DIM, 5, output, stride=2)
    output = Normalize(name + '.BN2', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Conv2D(name + '.3', 2*DIM, 4*DIM, 5, output, stride=2)
    output = Normalize(name + '.BN3', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Conv2D(name + '.4', 4*DIM, 8*DIM, 5, output, stride=2)
    output = Normalize(name + '.BN4', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = tf.reshape(output, [-1, 4*4*8*DIM])
    output = Linear(name + '.Output', 4*4*8*DIM, latent_dim, output)

    conv2d.unset_weights_stdev()
    deconv2d.unset_weights_stdev()
    linear.unset_weights_stdev()

    return output

# i.e. Generator
def dcgan_decoder(name, z, n_channels, is_training, mode=None, nonlinearity=tf.nn.relu):
    conv2d.set_weights_stdev(0.02)
    deconv2d.set_weights_stdev(0.02)
    linear.set_weights_stdev(0.02)

    output = Linear(name + '.Input', z.get_shape().as_list()[1], 4*4*8*DIM, z)
    output = tf.reshape(output, [-1, 8*DIM, 4, 4])
    output = Normalize(name + '.BN1', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Deconv2D(name +'.2', 8*DIM, 4*DIM, 5, output)
    output = Normalize(name + '.BN2', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Deconv2D(name +'.3', 4*DIM, 2*DIM, 5, output)
    output = Normalize(name + '.BN3', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Deconv2D(name +'.4', 2*DIM, DIM, 5, output)
    output = Normalize(name + '.BN4', [0,2,3], output, is_training, mode)
    output = nonlinearity(output)

    output = Deconv2D(name +'.5', DIM, n_channels, 5, output)
    output = tf.reshape(output, [-1, n_channels*DIM*DIM])

    conv2d.unset_weights_stdev()
    deconv2d.unset_weights_stdev()
    linear.unset_weights_stdev()

    return output

def NetsRetreiver(arch):
    if arch == 'low_cap':
        return low_capacity_encoder, low_capacity_decoder
    if arch == 'high_cap':
        return high_capacity_encoder, high_capacity_decoder
    if arch == 'resnet':
        return resnet_encoder, resnet_decoder
    if arch == 'dcgan':
        return dcgan_encoder, dcgan_decoder
    raise Exception()
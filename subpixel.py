from __future__ import division

import tensorflow as tf
from tensorflow.python.ops.init_ops import _compute_fans
from tensorflow.keras.initializers import Initializer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D


class ICNR(Initializer):

  def __init__(self, scale):
    self.scale = scale

  def __call__(self, shape, dtype=None):
    initializer = tf.keras.initializers.GlorotNormal()
    if self.scale == 1:
      return initializer(shape, dtype)
    else:
      new_shape = shape[:3] + (shape[3] // self.scale ** 2,)
      x = initializer(new_shape, dtype)
      x = tf.transpose(x, perm=[2, 0, 1, 3])
      x = tf.image.resize(x, size=(shape[0] * self.scale, shape[1] * self.scale), method='nearest')
      x = tf.compat.v1.space_to_depth(x, block_size=self.scale)
      x = tf.transpose(x, perm=[1, 2, 0, 3])
      return x

  def get_config(self):  # To support serialization
    return {"scale": self.scale}

# class ICNR(VarianceScaling):
#     """docstring for ICNR"""
#     def __init__(self, *args, **kwargs):
#         super(ICNR, self).__init__(*args, **kwargs)
#         #self.arg = arg

#     def __call__(self, shape, dtype=None, partition_info=None):
#         if dtype is None:
#           dtype = self.dtype
#         scale = self.scale
#         scale_shape = shape
#         if partition_info is not None:
#           scale_shape = partition_info.full_shape
#         fan_in, fan_out = _compute_fans(scale_shape)
#         if self.mode == "fan_in":
#           scale /= max(1., fan_in)
#         elif self.mode == "fan_out":
#           scale /= max(1., fan_out)
#         else:
#           scale /= max(1., (fan_in + fan_out) / 2.)
#         #if self.distribution == "normal" or self.distribution == "truncated_normal":
#           # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
#         stddev = tf.sqrt(scale) / .87962566103423978

#         #if self.scale == 1:
#         if scale == 1:
#             return tf.random.truncated_normal(shape, 0.0, stddev, dtype, seed=self.seed)

#         #logging.info(scale)

#         new_shape = shape[:3] + (shape[3] // (self.scale ** 2),)

#         x = tf.random.truncated_normal(new_shape, 0.0, stddev, dtype, seed=self.seed)
#         x = tf.transpose(x, perm=[2, 0, 1, 3])
#         x = tf.image.resize_nearest_neighbor(x, size=(shape[0] * self.scale, shape[1] * self.scale))
#         #x = tf.space_to_depth(x, block_size=scale)
#         x = tf.space_to_depth(x, block_size=self.scale)
#         x = tf.transpose(x, perm=[1, 2, 0, 3])

#         return x



'''
def icnr_weights(init = tf.glorot_normal_initializer(), scale=2, shape=[3,3,32,4], dtype = tf.float32):
    sess = tf.Session()
    return sess.run(ICNR(init, scale=scale)(shape=shape, dtype=dtype))

class ICNR:
    """ICNR initializer for checkerboard artifact free sub pixel convolution
    Ref:
     [1] Andrew Aitken et al. Checkerboard artifact free sub-pixel convolution
     https://arxiv.org/pdf/1707.02937.pdf)
    Args:
    initializer: initializer used for sub kernels (orthogonal, glorot uniform, etc.)
    scale: scale factor of sub pixel convolution
    """

    def __init__(self, initializer, scale=1):
        self.scale = scale
        self.initializer = initializer

    def __call__(self, shape, dtype, partition_info=None):
        shape = list(shape)
        if self.scale == 1:
            return self.initializer(shape)

        new_shape = shape[:3] + [shape[3] // (self.scale ** 2)]
        x = self.initializer(new_shape, dtype, partition_info)
        x = tf.transpose(x, perm=[2, 0, 1, 3])
        x = tf.image.resize_nearest_neighbor(x, size=(shape[0] * self.scale, shape[1] * self.scale))
        x = tf.space_to_depth(x, block_size=self.scale)
        x = tf.transpose(x, perm=[1, 2, 0, 3])

        return x
'''
class Subpixel(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 r,
                 padding='valid',
                 data_format=None,
                 strides=(1,1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel, self).__init__(
            filters=r*r*filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        bsize, a, b, c = I.get_shape().as_list()
        bsize = K.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = K.reshape(I, [bsize, a, b, int(c/(r*r)),r, r]) # bsize, a, b, c/(r*r), r, r
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)
        #Keras backend does not support tf.split, so in future versions this could be nicer
        X = [X[:,i,:,:,:,:] for i in range(a)] # a, [bsize, b, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:,i,:,:,:] for i in range(b)] # b, [bsize, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r*unshifted[1], self.r*unshifted[2], int(unshifted[3]/(self.r*self.r)))

    def get_config(self):
        config = super(Conv2D, self).get_config()
        #config.pop('rank')
        config.pop('dilation_rate')
        config['filters']= int(config['filters'] / self.r*self.r)
        config['r'] = self.r
        return config
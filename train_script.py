from __future__ import print_function

import tensorflow as tf
tf.enable_eager_execution()
#will not work without eager execution
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from deeplabv3p import Deeplabv3
import os

#if tf.__version__[0] == "2":
#    _IS_TF_2 = True
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LambdaCallback
#from tensorflow.keras.layers import *
from tensorflow.keras.layers import Reshape, Activation
from subpixel import ICNR, Subpixel
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

from utils import Jaccard, crossentropy_with_reshape
from tfrecord_iterator import parse_tfrecords

input_shape = (600, 600, 3)
num_classes = 6
backbone = 'xception'

losses = crossentropy_with_reshape
metrics = {'pred_mask' : [Jaccard]}


def get_callbacks(snapshot_every_epoch, snapshot_path, checkpoint_prefix):
    callbacks = []

    os.makedirs(snapshot_path, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(snapshot_path, str(checkpoint_prefix)+'_{epoch:02d}.h5'),
        verbose=1,
        period=snapshot_every_epoch)

    callbacks.append(checkpoint)

    return callbacks

def get_uncompiled_model(input_shape, num_classes, backbone, infer=False):

    model = Deeplabv3(weights=None, input_tensor=None, infer=infer,
                      input_shape=input_shape, classes=num_classes,
                      backbone=backbone, OS=16, alpha=1)


    base_model = Model(model.input, model.layers[-5].output)
    #self.net = net
    #modelpath = 'weights/{}_{}.h5'.format(backbone, net)

    if backbone=='xception':
        scale = 4
    else:
        scale = 8

    #elif net == 'subpixel':
    x = Subpixel(num_classes, 1, scale, padding='same')(base_model.output)
    #x = Reshape((input_shape[0]*input_shape[1], -1)) (x)
    #x = Reshape((input_shape[0]*input_shape[1], num_classes)) (x)
    # x = Activation('softmax', name = 'pred_mask')(x)
    model = Model(base_model.input, x, name='deeplabv3p_subpixel')

    # Do ICNR
    for layer in model.layers:
        if type(layer) == Subpixel:
            c, b = layer.get_weights()
            w = ICNR(scale=scale)(shape=c.shape)
            #W = tf.convert_to_tensor(w, dtype=tf.float32)
            layer.set_weights([w, b])

    return model


if __name__ == '__main__':

    model = get_uncompiled_model(input_shape, num_classes, backbone)
    model.load_weights('weights/{}_{}.h5'.format(backbone, 'subpixel'), by_name=True)
    #model.load_weights('/mnt/mydata/dataset/Playment_top_5_dataset/deeplab_top_5_classes_10.h5', by_name=True)

    #print(model.summary())

    model.compile(optimizer = Adam(lr=1e-3, epsilon=1e-8, decay=1e-6),
                  loss = crossentropy_with_reshape)#, metrics = metrics)

    input_function = parse_tfrecords(
        filenames='/mnt/mydata/dataset/Playment_top_5_dataset/test.tfrecords',
        height=600,
        width=600,
        num_classes=num_classes,
        batch_size=2)

    callbacks = get_callbacks(
        snapshot_every_epoch=5, 
        snapshot_path='/mnt/mydata/dataset/Playment_top_5_dataset', 
        checkpoint_prefix='deeplab_top_5_classes_focal_loss')


    model.fit(input_function, 
        epochs=15, 
        steps_per_epoch=346, 
        initial_epoch=0, 
        callbacks=callbacks)
    

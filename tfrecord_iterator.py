import tensorflow as tf
import glob, os, tqdm

def parse_tfrecords(filenames, height, width, num_classes, batch_size=32):

    def _parse_function(serialized):
        features = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/depth': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask/height': tf.FixedLenFeature([], tf.int64),
        'mask/width': tf.FixedLenFeature([], tf.int64),
        'mask/depth': tf.FixedLenFeature([], tf.int64),
        'mask_raw': tf.FixedLenFeature([], tf.string)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)

        image_string = parsed_example['image_raw']
        mask_string = parsed_example['mask_raw']
        #depth_string = parsed_example['depth_raw']

        # decode the raw bytes so it becomes a tensor with type

        image = tf.cast(tf.image.decode_jpeg(image_string), tf.uint8)
        image = tf.image.resize_images(image,(height, width))
        image.set_shape([height, width,3])

        mask = tf.cast(tf.image.decode_png(mask_string), tf.uint8)
        mask = tf.image.resize_images(mask,(height, width))
        mask.set_shape([height, width,1])

        #mask = tf.reshape(mask, shape=(height*width, 1))
        mask = tf.squeeze(tf.keras.backend.one_hot(tf.dtypes.cast(mask, tf.int32),num_classes))
        #mask = tf.keras.backend.one_hot(tf.squeeze(tf.dtypes.cast(mask, tf.int32)), num_classes) #[:,:,:-1]

        return tf.cast(image, tf.float32)/255 , mask
    
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(_parse_function, num_parallel_calls=4)

    dataset = dataset.shuffle(buffer_size=4)

    dataset = dataset.repeat(-1) # Repeat the dataset this time
    dataset = dataset.batch(batch_size)    # Batch Size
    batch_dataset = dataset.prefetch(buffer_size=4)

    #iterator = batch_dataset.make_one_shot_iterator()   # Make an iterator
    #batch_features,batch_labels = iterator.get_next()  # Tensors to get next batch of image and their labels
    
    #return batch_features, batch_labels
    return batch_dataset

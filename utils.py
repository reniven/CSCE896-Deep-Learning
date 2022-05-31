import numpy as np
import tensorflow as tf

def normalize_resize(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255)
    image = tf.image.resize(image, (28, 28))
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)
    return image, label 

def augmentation(image_size):
    
    resize_and_rescale = tf.keras.Sequential([
        tf.layers.experimental.preprocessing.Resizing(image_size[0], image_size[1]),
    ])

    data_augmentation = tf.keras.Sequential([
        tf.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.layers.experimental.preprocessing.RandomZoom(0.4),
        tf.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    return resize_and_rescale, data_augmentation

AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, shuffle=False, augment=False, batch_size=32):

    resize_and_rescale, data_augmentation = augmentation([36, 36])
    # Resize and rescale all datasets
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
              num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
            num_parallel_calls=AUTOTUNE)

    # Use buffered prefecting on all datasets
    return ds.prefetch(buffer_size=AUTOTUNE)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy import stats
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions

def getAlexNet(num_classes):
    model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(224,224,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(3, activation='softmax')
    ])
    return model

def getAlexNetFeat(model):
    ret_model = model
    ret_model.pop()
    ret_model.pop()
    return ret_model

def model(name):
    base_model = None
    if(name == 'vgg'):
        tmp_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3),
                                               include_top=True,
                                               weights='imagenet')
        base_model = Sequential()
        for layer in tmp_model.layers[:-1]:
            base_model.add(layer)
        base_model.add(Dense(3, activation='softmax'))
        
    if(name == 'alexnet'):
        base_model = getAlexNet(3)
    base_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                       loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                       metrics=['accuracy'])
    return base_mode

def extraction_model(name,base_model):
    if(name == 'vgg'):
        base_model.pop()
    if(name == 'alexnet'):
        base_model = getAlexNetFeat(base_model)
    return base_model

def feature_extraction(model, images,base_model):
    if(model == 'vgg'):
        base_model = extraction_model(model)
    else:
        base_model = extraction_model(model)
    img = image.load_img(images)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x)
    return features

def image_feature_creation(model, folders):
    
    for x in folders:
        features = feature_extraction(model, folders):
        return features

def data_transformer(image):

    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        featurewise_std_normalization=True,
        featurewise_center=True,
        horizontal_flip=True,
        vertical_flip=True)
    data_transformer = train_gen.flow_from_directory(
        batch_size=16,
        directory=image
        shuffle=True,
        target_size=(224, 224),
        class_mode= None)
        
    return data_transformer

def regression(data):
    alphas = np.logspace(-1, 2, 5, 7)
    best_score = 0

    for alpha in alphas:
        kFold = KFold(n_splilts = 10, shuffle = True)
        
        for train, test in kFold.split(data):
            regression = Ridge()

            train = data[train]
            test = data[test]
            x_train = train[:, :-1]
            y_train = train[:, -1]
            x_test = test[:, :-1]
            y_test = test[:, -1]

            pca = PCA(n_components=100)
            x_train = pca.fit(x_train)
            x_test = pca.fit(x_test)

            regression.fit(x_train, y_train)
            y_new = regression.predict(x_test)
            score = stats.pearsonr(y_test, y_new)
            score = score * score
            if(score > best_score):
                best_score = score
   
    return best_score

def plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

def load_data():

    # Training Data
    trainingImages = []
    trainingLabels = []

    path0 = 'cnn_images/train/0/'
    path1 = 'cnn_images/train/1/'
    path2 = 'cnn_images/train/2/'

    class_0_train = os.listdir(path_0)
    class_2_train = os.listdir(path_2)
    class_3_train = os.listdir(path_3)

    for i in [class_0_train, class_2_train, class_3_train]:
        for x in i:
            trainingLabels.append(0)
            if(i == class_0)
                trainingImages.append(path0_train + x)
            elif(i == class_1):
                trainingImages.append(path1_train + x)
            else:
                trainingImages.append(path2_train + x)

    # Testing Data
    testingImages = []
    testingImages = []
    path0_test  = 'cnn_images/valid/0/'
    path1_test  = 'cnn_images/valid/1/'
    path2_test  = 'cnn_images/valid/2/'

    class_0_test = os.listdir(path_0_test )
    class_2 = os.listdir(path_2_test_test  )
    class_3 = os.listdir(path_3_test_test  )

    for i in [class_0_test , class_2_test , class_3_test]:
        for x in i:
            trainingLabels.append(0)
            if(i == class_0)
                trainingImages.append(path0_test  + x)
            elif(i == class_1):
                trainingImages.append(path1_test  + x)
            else:
                trainingImages.append(path2_test  + x)

    return trainingLabels, trainingImages, testingLabels, testingImages

def load_country(country):

    data = None

    if("malawi"):
        data = np.loadtxt('features/malawi_features.csv')
    elif("nigeria"):
        data = np.loadtxt('features/nigeria.csv')
    else:
        data = np.loadtxt('features/ethiopia.csv')

    return data


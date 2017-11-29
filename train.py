'''
Created on July 25, 2017
@author: MarkVilla
'''
from __future__ import print_function
import os
from datetime import datetime
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from data import load_train_data, load_test_data, save_test_results

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 1024
img_cols = 1024
img_channels = 1
weights = 'a.h5'
train_group_size = 16
train_size = train_group_size*2
batch_size = 1
nb_epoch = 20
test_size = 1
smooth = 0.
lr = 1e-5


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
    #


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(data, size):
    data_p = np.ndarray((size, img_rows, img_cols), dtype=np.uint8)
    for i in range(size):
        data_p[i] = resize(data[i], (img_rows, img_cols), preserve_range=True)
        if (i+1) % 100 == 0:
            print('{0}/{1} preprocessed...'.format((i+1), size))
        elif (i+1) == (size):
            print('{0}/{1} preprocessed...'.format((i+1), size))

    data_p = data_p[..., np.newaxis]
    return data_p


def train_and_predict():

    print('-'*30)
    print('train info...')
    print('date = {0}'.format(datetime.now().strftime('%m/%d/%y %H:%M:%S')))
    print('input h,w,c = {0},{1},{2}'.format(img_rows, img_cols, img_channels))
    print('weights = {0}'.format(weights))
    print('train size = {0}*{1}'.format(train_group_size, train_size/train_group_size))
    print('nb epoch = {0}'.format(nb_epoch))
    print('batch size = {0}'.format(batch_size))
    print('test size = {0}'.format(test_size))
    print('-'*30)

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    train_images, train_masks = load_train_data()
    train_images = preprocess(train_images, train_size)
    train_masks = preprocess(train_masks, train_size)
    train_images = train_images.astype('float32')
    mean = np.mean(train_images)  # mean for data centering
    std = np.std(train_images)  # std for data normalization
    train_images -= mean
    train_images /= std
    train_masks = train_masks.astype('float32')
    train_masks /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights(weights) # keep this commented if no pretrained waights are needed
    model_checkpoint = ModelCheckpoint(weights, monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(train_images, train_masks, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True, validation_split=0.2, callbacks=[model_checkpoint])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    test_images, test_image_ids = load_test_data()
    test_images = preprocess(test_images, test_size)
    test_images = test_images.astype('float32')
    test_images -= mean
    test_images /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights(weights)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    test_masks = model.predict(test_images, verbose=1)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    save_test_results(test_masks, test_image_ids)


if __name__ == '__main__':
    train_and_predict()

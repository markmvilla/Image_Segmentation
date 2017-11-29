'''
Created on July 25, 2017
@author: MarkVilla
'''
from __future__ import print_function
import os
import numpy as np
from skimage import img_as_ubyte
from skimage.io import imsave, imread

data_path = 'data/'

image_cols = 1918
image_rows = 1280

def create_train_data():
    image_train_path = os.path.join(data_path, 'train_images')
    mask_train_path = os.path.join(data_path, 'train_masks')
    train_image_list = os.listdir(image_train_path)
    total = len(train_image_list)

    images = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    masks = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    print('-'*30)
    print('Loading training images...')
    print('-'*30)
    for i, image_name in enumerate(train_image_list):
        mask_name = image_name.split('.')[0] + '_mask.gif'
        image = imread(os.path.join(image_train_path, image_name), as_grey=True)
        mask = imread(os.path.join(mask_train_path, mask_name), as_grey=True)

        image = img_as_ubyte(np.array([image]))
        mask = img_as_ubyte(np.array([mask]))
        images[i] = image
        masks[i] = mask

        if (i+1) % 100 == 0:
            print('{0}/{1} images loaded.'.format((i+1), total))
        elif (i+1) == (total):
            print('{0}/{1} images loaded.'.format((i+1), total))

    print('Loading done.')

    np.save('data/np/train_images.npy', images)
    np.save('data/np/train_masks.npy', masks)
    print('Saving to .npy files done.')


def load_train_data():
    train_images = np.load('data/np/train_images.npy')
    print('Loaded train images.')
    train_masks = np.load('data/np/train_masks.npy')
    print('Loaded train masks.')
    return train_images, train_masks


def create_test_data():
    image_test_path = os.path.join(data_path, 'test_images')
    test_image_list = os.listdir(image_test_path)
    total = len(test_image_list)

    images = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    image_ids = np.ndarray((total, ), dtype=object)

    print('-'*30)
    print('Loading test images...')
    print('-'*30)
    for i, image_name in enumerate(test_image_list):
        image_id = str(image_name.split('.')[0])
        image = imread(os.path.join(image_test_path, image_name), as_grey=True)

        image = img_as_ubyte(np.array([image]))
        images[i] = image
        image_ids[i] = image_id

        if (i+1) % 100 == 0:
            print('{0}/{1} images loaded.'.format((i+1), total))
        elif (i+1) == (total):
            print('{0}/{1} images loaded.'.format((i+1), total))

    print('Loading done.')

    np.save('data/np/test_images.npy', images)
    np.save('data/np/test_image_ids.npy', image_ids)
    print('Saving to .npy files done.')


def load_test_data():
    test_images = np.load('data/np/test_images.npy')
    print('Loaded test images.')
    test_image_ids = np.load('data/np/test_image_ids.npy')
    print('Loaded test image_ids.')
    return test_images, test_image_ids


def save_test_results(test_masks, test_image_ids):
    np.save('data/np/test_masks.npy', test_masks)
    for image, image_id in zip(test_masks, test_image_ids):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(data_path, 'test_masks', str(image_id) + '_pred.png'), image)


if __name__ == '__main__':
    create_train_data()
    create_test_data()

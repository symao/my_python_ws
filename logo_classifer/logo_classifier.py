from __future__ import print_function
import cv2
import os
import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential,model_from_json,Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras.optimizers import RMSprop, SGD, Adam, Adadelta

img_rows, img_cols = 28, 28
num_classes = 2

def load_data():
    posdir = 'data/pos'
    pos_imgs = [cv2.resize(cv2.imread(os.path.join(posdir,f), cv2.IMREAD_GRAYSCALE),(img_cols,img_rows)) for f in sorted(os.listdir(posdir))]
    negdir = 'data/neg'
    neg_imgs = [cv2.resize(cv2.imread(os.path.join(negdir,f), cv2.IMREAD_GRAYSCALE),(img_cols,img_rows)) for f in sorted(os.listdir(negdir))]
    negdir2 = '/home/symao/data/COCO/test2017'
    neg_imgs += [cv2.resize(cv2.imread(os.path.join(negdir2,f), cv2.IMREAD_GRAYSCALE),(img_cols,img_rows)) for f in sorted(os.listdir(negdir2))[:len(pos_imgs)*4-len(neg_imgs)]]

    pos_cut = int(len(pos_imgs)*0.9)
    neg_cut = int(len(neg_imgs)*0.9)
    x_train = np.array(pos_imgs[:pos_cut]+neg_imgs[:neg_cut])
    x_test = np.array(pos_imgs[pos_cut:]+neg_imgs[neg_cut:])

    y_train = np.array([1]*pos_cut+[0]*neg_cut)
    y_test = np.array([1]*(len(pos_imgs)-pos_cut) + [0]*(len(neg_imgs)-neg_cut))
    return (x_train,y_train), (x_test, y_test)

def cnn_model(input_shape,num_classes):
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
          optimizer='Adadelta',
          metrics=['accuracy'])
    return model

def cnn_train_test(exist_model = None):
    batch_size = 128
    epochs = 100
    (x_train, y_train), (x_test, y_test) = load_data()

    if keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if exist_model is None:
        model = cnn_model(input_shape,num_classes)
    else:
        model = model_from_json(open(exist_model+'.json').read())
        model.load_weights(exist_model+'.h5')
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adadelta', metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=True,
              validation_data=(x_test, y_test))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=True)
    print('CNN Test loss:%f Accuracy:%f'%(loss, accuracy))

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # model.save(os.path.join(save_dir, 'logo_classifier.hdf5'))
    with open(os.path.join(save_dir, 'logo_classifier.json'), 'w') as fout:
        fout.write(model.to_json())
    model.save_weights(os.path.join(save_dir, 'logo_classifier.h5'), overwrite=True)

def test_model():
    img = cv2.resize(cv2.imread('/home/symao/a.jpg',cv2.IMREAD_GRAYSCALE), (img_cols,img_rows))
    if keras.backend.image_data_format() == 'channels_first':
        r,c = img.shape
        img = img.reshape(1, 1, r, c)
        input_shape = (1, img_rows, img_cols)
    else:
        r,c = img.shape
        img = img.reshape(1, r, c, 1)
        input_shape = (img_rows, img_cols, 1)
    img = img.astype('float32')
    img /= 255
    # model = cnn_model(input_shape, num_classes)
    model = model_from_json(open('./saved_models/logo_classifier.json').read())
    model.load_weights('./saved_models/logo_classifier.h5')

    # intermediate_layer_model = Model(input=model.input,
    #                              output=model.get_layer('dense_1').output)
    # out = intermediate_layer_model.predict(img)
    out = model.predict(img)
    print(out.shape)
    print(out)


if __name__ == '__main__':
    # cnn_train_test()
    cnn_train_test('saved_models/logo_classifier')
    # test_model()
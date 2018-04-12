import numpy as np
import cv2
import os
from math import *

def rotate_image(img, angle_deg):
    h,w = img.shape[:2]
    sa = sin(radians(angle_deg))
    ca = cos(radians(angle_deg))
    hnew = w*fabs(sa)+h*fabs(ca)
    wnew = h*fabs(sa)+w*fabs(ca)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1)
    mask = 255 - cv2.warpAffine(np.zeros(img.shape,img.dtype)+255,M,(w,h))
    rotated_img = cv2.add(cv2.warpAffine(img, M, (w,h)), mask)
    return rotated_img

def image_augment(input_img, N):
    img_list = [rotate_image(input_img, angle) for angle in np.random.uniform(0,360,N)]
    for img in img_list:
        if np.random.random()<0.5:#illumination
            r = np.random.random()
            if r<0.3:
                M = np.ones(img.shape,dtype='uint8')*np.random.randint(30,60)
            elif r<0.6:
                M = np.ones(img.shape,dtype='uint8')*np.random.randint(60,100)
            elif r<0.8:
                cols = img.shape[1]
                a = np.ones(img.shape[1:],dtype='uint8')
                b = (np.linspace(np.random.randint(0,10), np.random.randint(30,60),cols)).astype('uint8')
                M = np.ones(img.shape,dtype='uint8')*((a.T*b).T)
            else:
                cols = img.shape[1]
                a = np.ones(img.shape[1:],dtype='uint8')
                b = (np.linspace(np.random.randint(0,10), np.random.randint(0,30),cols)).astype('uint8')
                M = np.ones(img.shape,dtype='uint8')*((a.T*b).T)
            if np.random.random()<0.5:
                img = cv2.add(img,M)
            else:
                img = cv2.subtract(img,M)
        if np.random.random()<0.4:#blur
            size = np.random.choice([3,5])
            if np.random.random()<0.4:
                img = cv2.GaussianBlur(img,(size,size),np.random.uniform(0.2,1.5))
            else:
                img = cv2.medianBlur(img,size)
        if np.random.random()<0.2:#noise
            if np.random.random()<0.4: #crop
                crop_size = np.random.randint(1, 5)
                amount = np.random.randint(1, 3)
                coords = [np.random.randint(0, i, amount) for i in img.shape]
                for i in range(amount):
                    x = coords[0][i]
                    y = coords[1][i]
                    img[x:x+crop_size,y:y+crop_size] = 0
            if np.random.random()<0.3: #salt pepper
                s_vs_p = 0.3
                amount = 0.0004
                # Salt mode
                num_salt = np.ceil(amount * img.size * s_vs_p)
                coords = [np.random.randint(0, i, int(num_salt)) for i in img.shape]
                img[coords] = 255
                # Pepper mode
                num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
                coords = [np.random.randint(0, i, int(num_pepper)) for i in img.shape]
                img[coords] = 0
            if np.random.random()<0.3: #gaussian
                amount = np.random.random()*0.4
                mean = 0
                var = 20
                sigma = var**0.5
                num_gauss = np.ceil(amount * img.size)
                gauss = np.random.normal(mean,sigma,img.shape)
                coords = [np.random.randint(0, i, int(num_gauss)) for i in img.shape]
                img[coords] += gauss[coords].astype('uint8')
    return img_list

def data_augment(rawdir, savedir, imsize):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    fimgs = [os.path.join(rawdir,f) for f in os.listdir(rawdir)]

    img_list = []
    for fimg in fimgs:
        img = cv2.resize(cv2.imread(fimg, cv2.IMREAD_GRAYSCALE),imsize)
        img_list += image_augment(img, 15)
    for i,img in enumerate(img_list):
        cv2.imwrite(os.path.join(savedir,'%06d.jpg'%i),img)


def build_pos():
    rawdir = 'data/pos_raw'
    savedir = 'data/pos'
    imgsize = (28,28)
    data_augment(rawdir,savedir,imgsize)
    with open('data/pos.txt','w') as fp:
        for f in os.listdir(savedir):
            fp.write(os.path.join(savedir,f)+' 1 0 0 %d %d\n'%imgsize)

def build_neg():
    rawdir = 'data/neg_raw'
    savedir = 'data/neg'
    imgsize = (28,28)
    data_augment(rawdir,savedir,imgsize)
    with open('data/neg.txt','w') as fp:
        for f in os.listdir(savedir):
            fp.write(os.path.join(savedir,f))


if __name__ == '__main__':
    build_pos()
    build_neg()

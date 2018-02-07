#!/usr/bin/python2.7
import cv2
import numpy as np
import os
import sys
import argparse
from video_cut import grab_video_from_video
from video_downsample import video_dowmsample
from video_convert import video_convert
from video_resize import video_resize
from video_combine import video_combine
from video_compose import video_compose
from video2img import video2img

def get_input(info,default,default_type):
    t = raw_input(info)
    if t == '':
        return default_type(default)
    else:
        return default_type(t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video Process Tool")
    parser.add_argument("input_video")
    parser.add_argument('-o', '--output', help='output file')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--cut', action='store_true', help='open video cut or crop ui')
    group.add_argument('-v', '--convert', action='store_true', help='video format convert')
    group.add_argument('-d', '--downsample', type=int, help='select 1/X frame')
    group.add_argument('-r', '--resize', type=float, help='image resize rate')
    group.add_argument('-a', '--append', help='append video')
    group.add_argument('-i', '--insert', help='insert video')
    group.add_argument('-t', '--toimages', help='image dir')

    args = parser.parse_args()
    if args.cut:
        print('=========== video cut/crop ===============')
        grab_video_from_video(args.input_video)
    elif args.toimages:
        print('=========== video to images ==============')
        stride = get_input('input stride[1]:',1,int)
        resize = get_input('input resize rate[1]:',1,float)
        suffix = get_input('input suffix[.png]','.png',str)
        video2img(args.input_video, args.toimages, stride, resize, suffix)
    elif not args.output:
        print('Please set output file with -o')
    elif args.downsample:
        print('=========== video downsample ==============')
        video_dowmsample(args.input_video, args.downsample, args.output)
    elif args.convert:
        print('=========== video format convert ==========')
        video_convert(args.input_video, args.output)
    elif args.resize:
        print('=========== video resize ==================')
        video_resize(args.input_video, args.resize, args.output)
    elif args.append:
        print('=========== video append ==================')
        video_combine(args.input_video, args.append, args.output)
    elif args.insert:
        print('=========== video insert ==================')
        video_compose(args.input_video, args.insert, args.output)


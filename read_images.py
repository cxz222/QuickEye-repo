# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:19:14 2019

@author: zcyhi
"""

import os
import open3d
import cv2
from imageio import imread
from PIL import Image
import numpy as np
import pickle

os.chdir("C:/Users/zcyhi/Documents/Insight Project/")

# Read PLY images and save 2D topview projection
for i in range(1,31):
    vis = open3d.visualization.Visualizer()
    print(i)
    vis.create_window(width = 400, height = 400)
    vis.add_geometry(open3d.io.read_point_cloud(
            'models_cad/obj_' + str(i).zfill(2) + '.ply'))
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image('Images/obj_' + str(i).zfill(2) + '.png')
    vis.destroy_window()

# Image for Object 24 needs to be rotated
# The code below generates the point cloud image of Object 24
# Image is then rotated and saved manually
pcd = open3d.io.read_point_cloud("models_cad/obj_24.ply")
vis = open3d.visualization.Visualizer()
vis.create_window(width = 400, height = 400)
vis.add_geometry(pcd)
vis.run()
vis.destroy_window()
    
# Convert the PNG for Object 24 to RGB
image24 = Image.open("Images/obj_24.png").convert('RGB')
image24.save("Images/obj_24.png")

# Resize all images to size 84 x 84
# To be consistent with miniImageNet
for i in range(1,31):
    image = Image.open('Images/obj_' + str(i).zfill(2) + '.png')
    image84 = image.resize((84, 84), Image.ANTIALIAS)
    image84.save("Images/resized/obj_" + str(i).zfill(2) + ".png")
    
# Generate pickled file as input for meta-testing
all_img = np.empty((0, 84, 84, 3), dtype = np.uint8)
for i in range(1,31):
    img = imread("Images/resized/obj_" + str(i).zfill(2) + ".png")
    img = img.reshape(1,84,84,3)
    all_img = np.concatenate((all_img, img), axis = 0)
    print(i)
    print(all_img.shape)
data = np.concatenate((all_img[0:26,:,:,:], all_img[[27, 29],:,:,:]), axis = 0)

labels = list(np.concatenate((np.repeat(101,4), np.repeat(102,4),
                        np.repeat(103,4), np.repeat(104,6),
                        np.repeat(105,5), np.repeat(106,5))))
label2catname = ['Cap','Junction','Misc','Cylinder','Plugin','Nipple_box']
catname2label = {'Cap':101, 'Junction':102, 'Misc':103, 'Cylinder':104,\
                 'Plugin':105, 'Nipple_box':106}
obj_dict = {'catname2label': catname2label,
            'data': data,
            'labels':labels,
            'label2catname': label2catname}

outfile = open('miniImageNet_category_split_test.pickle','wb')
pickle.dump(obj_dict, outfile, protocol=2)
outfile.close()
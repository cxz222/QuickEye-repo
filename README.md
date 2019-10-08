# QuickEye: Ientify 3D Objects with Few-Shot Learning

This project aims at building classifiers for 3-D images from the T-LESS dataset. Specifically, we labeled 28 images from T-LESS CAD models into 6 disjoint categories (classes).

To meet the challenge of the small sample, this project employs two distinct approaches:

![alt text](https://raw.githubusercontent.com/cxz222/QuickEye-repo/master/README%20images/strategies.png)

1. Augmenting the data by taking 8 2-D snapshots from different angles for each object. These 2-D images are then classified using the following steps:
* Take 2-D snapshots with different angles and random field of views **(render mesh.R)**
* Identify keypoints and extract features using SIFT **(bag_of_visual_words.py)**
* Use k-means clustering to construct codebook of visual words **(bag_of_visual_words.py)**
* Create a histogram to represent each image, and perform PCA to reduce dimensions of the features **(bag_of_visual_words.py)**
* Build 6-way classifier **(bag_of_visual_words.py)**

2. Apply the MetaOptNet meta-learning method on topview 2-D projections of the 3-D objects (point clouds).
* Obtain 2-D topview projections using 400 x 400 frames, then resize images into 84 x 84 RBG files in the PNG format **(read_images.py)**
* Pack all data, labels, and category (labels) information into pickle file **(read_images.py)**
* Apply MetaOptNet **(apply MetaOptNet.ipynb)**
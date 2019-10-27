# QuickEye: Ientify 3D Objects with Few-Shot Learning

This project aims at building classifiers for 3-D images from the T-LESS dataset. Specifically, we labeled 28 images from T-LESS CAD models into 6 disjoint categories (classes).

#### [My Medium story on this project](https://blog.insightdatascience.com/can-we-identify-3-d-images-using-very-little-training-data-ebae1ed1d8f9?source=friends_link&sk=701069b966d51c67e48b75854e507d9c) documents the technical considerations in more detail
#### A slide deck summarizing this project can be found in this [link](https://docs.google.com/presentation/d/1gPI8mBHrRALxYvreQ2w6GE0USxO4yxoAEVMmDMSYpK4/edit?usp=sharing)


## Pipelines

I developed two distinct pipelines for this task. The first one follows the logic of data augmentation and feed multiple images per object into a bag of visual words model. The second one takes only one snapshot of each object and applies a meta-learning method documented here: https://github.com/kjunelee/MetaOptNet

![alt text](https://raw.githubusercontent.com/cxz222/QuickEye-repo/master/README%20images/pipeline.png)


#### 1. Augmenting the data by taking 8 2-D snapshots from different angles for each object. These 2-D images are then classified using the following steps:
* Take 2-D snapshots with different angles and random field of views **(render mesh.R)**
* Identify keypoints and extract features using SIFT **(bag_of_visual_words.py)**
* Use k-means clustering to construct codebook of visual words **(bag_of_visual_words.py)**
* Create a histogram to represent each image, and perform PCA to reduce dimensions of the features **(bag_of_visual_words.py)**
* Build 6-way classifier **(bag_of_visual_words.py)**

#### 2. Apply the MetaOptNet meta-learning method on topview 2-D projections of the 3-D objects (point clouds).
* Obtain 2-D topview projections using 400 x 400 frames, then resize images into 84 x 84 RBG files in the PNG format **(read_images.py)**
* Pack all data, labels, and category (labels) information into pickle file **(read_images.py)**
* Apply MetaOptNet **(protonet on 2-D t-less.ipynb)**

![alt text](https://raw.githubusercontent.com/cxz222/QuickEye-repo/master/README%20images/meta-learning.png)
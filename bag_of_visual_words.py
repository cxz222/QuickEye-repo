import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold

os.chdir("C:/Users/zcyhi/Documents/Insight Project/rotated images/Exp 1")

#-------------------------------Define Functions-------------------------------

# Obtain SIFT descriptors from all images and combine into one single array with 128 columns
# Train a k-means clustering model on all keypoints/descriptors
# Return a k-means clustering model trained on all images
# Each cluster centroid represents one visual word
def read_and_clusterize(file_images, num_cluster):

    sift_keypoints = []

    with open(file_images) as f:
        images_names = f.readlines()
        images_names = [a.strip() for a in images_names]

        for line in images_names:
            #read image
            image = cv2.imread(line,1)
            # Convert them to grayscale
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # SIFT extraction
            sift = cv2.xfeatures2d.SIFT_create()
            kp, descriptors = sift.detectAndCompute(image,None)
            #append the descriptors to a list of descriptors
            sift_keypoints.append(descriptors)

    sift_keypoints=np.asarray(sift_keypoints)
    sift_keypoints=np.concatenate(sift_keypoints, axis=0)

    print("Training kmeans")    
    kmeans = MiniBatchKMeans(n_clusters=num_cluster,
                             init_size=3*num_cluster,
                             random_state=0).fit(sift_keypoints)

    return kmeans

# Assign labels to images based on image number
def define_class(img_name):
    img_num = img_name[4:6]
    if img_name[4:6] in ['01','02','03','04']:
        class_image = 1
    elif img_name[4:6] in ['05','06','07','08']:
        class_image = 2
    elif img_name[4:6] in ['09','10','11','12']:
        class_image = 3
    elif img_name[4:6] in ['13','14','15','16','17','18']:
        class_image = 4
    elif img_name[4:6] in ['19','20','21','22','23']:
        class_image = 5
    else:
        class_image = 6
    return class_image, img_num

# Obtain SIFT descriptors from single images without combining
# Assign a cluster to each keypoint/descriptor using model stored in the "model" argument 
# Generate a histogram of words that represents an image
# Return respective arrays for class label, feature, and image number for each image (row)
def calculate_centroids_histogram(file_images, model, num_bins):

    feature_vectors=[]
    class_vectors=[]
    obj_vectors=[]

    with open(file_images) as f:
        images_names = f.readlines()
        images_names = [a.strip() for a in images_names]

        for line in images_names:
            image = cv2.imread(line,1)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kp, descriptors = sift.detectAndCompute(image,None)
            
            #assign a cluster to a keypoint/descriptor
            predict_kmeans=model.predict(descriptors)
            #calculates the histogram
            hist, bin_edges=np.histogram(predict_kmeans, bins=num_bins)
            #histogram is the feature vector
            feature_vectors.append(hist)
            #define the class of the image
            [class_sample, obj_num]=define_class(line)
            class_vectors.append(class_sample)
            obj_vectors.append(obj_num)

    feature_vectors=np.asarray(feature_vectors)
    class_vectors=np.asarray(class_vectors)
    obj_vectors=np.asarray(obj_vectors)
    #return vectors and classes we want to classify
    return class_vectors, feature_vectors, obj_vectors

# Return 
def reduce_dimension_pca(input_df, num_pc):
    pca = PCA(n_components = num_pc)
    principalComponents = pca.fit_transform(input_df)
    principalDf = pd.DataFrame(data = principalComponents)
    return(principalDf)

def analytic_df_by_param(num_cluster, num_bins, num_pc):
    model = read_and_clusterize('rotate_image_list.txt', num_cluster)
    [classvec, featvec, objvec] = calculate_centroids_histogram("rotate_image_list.txt", model, num_bins)
    normalized_df=(featvec-featvec.min())/(featvec.max()-featvec.min())
    analytic_df = reduce_dimension_pca(normalized_df, num_pc)
    analytic_df['Class'] = classvec
    analytic_df['Obj'] = objvec
    return(analytic_df)

def avg_accuracy(num_cluster, num_bins, num_pc):
    
    print("{} words, {} histogram bins, {} principal components".format(num_cluster, num_bins, num_pc))
    
    # Generate analytic dataset
    analytic_df = analytic_df_by_param(num_cluster, num_bins, num_pc)
    analytic_df = pd.merge(analytic_df, dims, on='Obj')
    
    # Hold-out test set
    df_test = analytic_df[analytic_df['Obj'].isin(testvec)]
    df_test = df_test.reset_index(drop = True)
    X_test = df_test.drop(['Class', 'Obj'], axis = 1).to_numpy()
    y_test = df_test['Class']

    df_tnt = analytic_df[~analytic_df['Obj'].isin(testvec)]
    df_tnt = df_tnt.reset_index(drop = True)

    accuracy = []
    for i in range(0,5):
        df_val = df_tnt[df_tnt['Obj'].isin(val_seq[:,i])]
        X_val = df_val.drop(['Class', 'Obj'], axis = 1).to_numpy()
        y_val = df_val['Class']
    
        df_train = df_tnt[~df_tnt['Obj'].isin(val_seq[:,i])]
        X_train = df_train.drop(['Class', 'Obj'], axis = 1).to_numpy()
        y_train = df_train['Class']        
        
        svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
        svm_predictions = svm_model_linear.predict(X_val)
        accuracy.append(svm_model_linear.score(X_val, y_val))

    print("Pseudo cross-validation accuracy: {:.1%}".format(np.mean(accuracy)))

    
def final_model_fit(num_cluster, num_bins, num_pc):
    # Generate analytic dataset
    analytic_df = analytic_df_by_param(num_cluster, num_bins, num_pc)
    analytic_df = pd.merge(analytic_df, dims, on='Obj')
    accuracy_test_list = []
    
    for class_omit in range(1,7):
        # Remove one class for 5-way classification
        analytic_df_5way = analytic_df.loc[analytic_df['Class'] != class_omit]
        
        # 2-way partition
        df_test = analytic_df_5way[analytic_df_5way['Obj'].isin(testvec)]
        df_test = df_test.reset_index(drop = True)
        X_test = df_test.drop(['Class', 'Obj'], axis = 1).to_numpy()
        y_test = df_test['Class']
    
        df_tnt = analytic_df_5way[~analytic_df_5way['Obj'].isin(testvec)]
        df_tnt = df_tnt.reset_index(drop = True)
        X_tnt = df_tnt.drop(['Class', 'Obj'], axis = 1).to_numpy()
        y_tnt = df_tnt['Class']
        
        svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_tnt, y_tnt) 
        svm_predictions = svm_model_linear.predict(X_tnt)
        accuracy = svm_model_linear.score(X_tnt, y_tnt)
        svm_predictions_test = svm_model_linear.predict(X_test)
        accuracy_test = svm_model_linear.score(X_test, y_test)
    
        print("Confusion matrix:")
        print(confusion_matrix(y_test, svm_predictions_test))
        print("Accuracy on test set: {:.1%}".format(accuracy_test))
        accuracy_test_list.append(accuracy_test)
        
    print("Average accuracy on test set: {:.1%}".format(np.mean(accuracy_test_list)))

#-------------------------------Main Script----------------------------------


# Generate list of images
files = []
for (dirpath, dirnames, filenames) in os.walk(os.getcwd()):
    files.extend(filenames)
    break
files = [x for x in files if (('27' in x) == False and
                              ('29' in x) == False and
                              ('obj' in x) == True)]

with open('rotate_image_list.txt', 'w') as txtfile:
    for item in files:
        txtfile.write("%s\n" % item)
        
# Add dimension/size information
dims = pd.read_csv("dimensions.csv") 
dims["Obj"] = dims['obj_num'].apply(lambda x: '{0:0>2}'.format(x))
dims = dims.drop('obj_num', axis = 1)
cols_to_norm = ['size_x','size_y','size_z']
dims[cols_to_norm] = dims[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Specify objects for hold-out test set
testvec = ['04','08','12','18','23','30']

# Randomly sample 5 classes for 5-way classification
sample_classes = random.sample(population=[1,2,3,4,5,6], k=5)

# Randomly generate 5-fold cross-validation partitions
val_seq = np.stack(
        (np.tile(np.random.choice(['01','02','03'], 3, replace=False), 2)[0:5],
         np.tile(np.random.choice(['05','06','07'], 3, replace=False), 2)[0:5],
         np.tile(np.random.choice(['09','10','11'], 3, replace=False), 2)[0:5],
         np.random.choice(['13','14','15','16','17'], 5, replace=False),
         np.tile(np.random.choice(['19','20','21','22'], 4, replace=False), 2)[0:5],
         np.tile(np.random.choice(['24','25','26','28'], 4, replace=False), 2)[0:5]),
    axis = 0)

# Test some hyperparameters
avg_accuracy(400, 20, 15)
avg_accuracy(400, 30, 15)
avg_accuracy(400, 40, 15)
avg_accuracy(625, 25, 15)
avg_accuracy(400, 20, 10)    # Winning model
avg_accuracy(900, 30, 15)
avg_accuracy(225, 15, 15)

# Obtain final prediction and accuracy on hold-out test set
final_model_fit(400, 20, 10)
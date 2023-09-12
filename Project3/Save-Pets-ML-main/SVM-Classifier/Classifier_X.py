import os
import numpy as np
import cv2

import pickle
import sklearn
import argparse
import time

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from histo_clahe import histo_clahe

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--dir', default='Dog-Data',help='dataset directory')
parser.add_argument('--test', default='test_1.jpg',help='test image data')
opt = parser.parse_args()

#read data
def read_data(label2id):
    X = []  # List to store image data
    Y = []  # List to store labels
    
    # Loop through the subdirectories in the 'Dog-Data/train' directory
    for label in os.listdir('C:/Users/Playdata/Desktop/Save-Pets-ML-main/SVM-Classifier/Dog-Data/train'):
        # Check if the item in 'Dog-Data/train' is a directory
        if os.path.isdir(os.path.join('C:/Users/Playdata/Desktop/Save-Pets-ML-main/SVM-Classifier/Dog-Data/train', label)):
            # Loop through image files in the current subdirectory
            for img_file in os.listdir(os.path.join('C:/Users/Playdata/Desktop/Save-Pets-ML-main/SVM-Classifier/Dog-Data/train', label)):
                # Read the image using OpenCV's imread function
                img = cv2.imread(os.path.join('C:/Users/Playdata/Desktop/Save-Pets-ML-main/SVM-Classifier/Dog-Data/train', label, img_file))
                
                # Append the image data to the X list
                X.append(img)
                
                # Use the label2id dictionary to map the label to an ID and append to Y list
                Y.append(label2id[label])
    
    # Return the lists of image data and labels
    return X, Y


# feature exraction with SIFT
def extract_sift_features(X):
    image_descriptors = []  # List to store SIFT descriptors for each image
    
    '''
    A SIFT (Scale-Invariant Feature Transform) descriptor is a vector 
    that encodes information about the local appearance and orientation
    of a keypoint in an image. 
    It summarizes the distribution of gradient directions 
    around the keypoint, making it robust to changes 
    in scale, rotation, and lighting conditions. 
    Each SIFT descriptor is computed for a specific keypoint 
    and is used to represent that keypoint's local image region.
    '''
    
    # Create a SIFT object with specific parameters
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=200, nOctaveLayers=3, contrastThreshold=0.0005)

    # Loop through the input images
    for i in range(len(X)):
        # Detect keypoints and compute descriptors using the SIFT object
        kp, des = sift.detectAndCompute(X[i], None)
        
        # Append the computed descriptors to the list of image_descriptors
        image_descriptors.append(des)

    # Return the list of SIFT descriptors for each image
    return image_descriptors


# Kmeans bow
def kmeans_bow(all_descriptors, num_clusters):
    bow_dict = []  # List to store cluster centers (visual words)
    
    # Perform K-means clustering on all SIFT descriptors to create the visual vocabulary
    kmeans = KMeans(n_clusters=num_clusters).fit(all_descriptors)
    
    # Get the cluster centers (visual words) from the K-means model
    bow_dict = kmeans.cluster_centers_
    
    return bow_dict


#image to vector.
def create_features_bow(image_descriptors, BoW, num_clusters):
    X_features = []  # List to store BoW features for each image
                     # BoW :  Bag of visual words (cluster centers obtained from K-means)
    
    for i in range(len(image_descriptors)):
        features = np.array([0] * num_clusters)  # Initialize a feature vector
        
        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)  # Compute distances to visual words
            argmin = np.argmin(distance, axis=1)  # Get the nearest visual word index for each descriptor
            
            # Count the occurrences of each visual word in the image
            for j in argmin:
                features[j] += 1
        
        X_features.append(features)  # Append the feature vector for the image
        
    return X_features

'''
This function converts the SIFT descriptors of an image into a BoW feature vector. 
It counts how many times each visual word appears in the image's descriptors and constructs a histogram-like vector.
Each element in the feature vector corresponds to the count of a visual word.

The combination of these two functions(kmeans_bow, create_features_bow) is fundamental 
to creating a BoW representation of images, which can be used as input features for machine learning classifiers.
'''

'''
---------------<Summary>---------------
defined 4 functions : read_data, extract_sift_features, kmeans_bow, create_features_bow

1. read_data(label2id) --> X, Y. 
    input - label2id = a dictionary containig labels and numerical identifiers
    output - X = List to store image data
             Y = List to store labels
             
2. extract_sift_features(X)  --> image_descriptors
    input - X = X from read_data() : the list to store image data
    outut - image_descriptors = the list of SIFT descriptors for each image
    
3. kmeans_bow(all_descriptors, num_clusters) --> bow_dict
    input - all_descriptors = the list containing all the image_descriptors
            num_clusters : the number of clusters the user defines
    output - bow_dict = a "list" containing cluster centers obtained from K-means 
    
4. create_features_bow(image_descriptors, BoW, num_clusters) --> X_features
    input - image_descriptors = extract_sift_features(X)
            BoW = kmeans_bow(all_descriptors, num_clusters)
            num_clusters : the number of clusters the user defines
    output - X_features : a list of BoW features for each image. 
                          Bag of visual words (cluster centers obtained from K-means)
'''


def main():
    start = time.time()
    # Label to id
    path = "C:/Users/Playdata/Desktop/Save-Pets-ML-main/SVM-Classifier/Dog-Data/train"
    file_list = os.listdir(path)
    label2id = {}

    idx = 0
    for i, label in enumerate(file_list):
        if label == ".DS_Store": 
            continue;
        # If the label is not ".DS_Store," 
        # a numerical identifier idx is assigned to that label,
        # and the label-ID mapping is stored in the label2id dictionary.
        label2id[label] = idx
        idx += 1
        
    X, Y = read_data(label2id)

    image_descriptors = extract_sift_features(X)

    # all descriptors
    all_descriptors = []
    for descriptors in image_descriptors:
        if descriptors is not None:
            for des in descriptors:
                all_descriptors.append(des)


    num_clusters = 100

    # saving the bow_dict(output from kmeans_bow function) to a pickle file & load th file
    if not os.path.isfile('C:/Users/Playdata/Desktop/Save-Pets-ML-main/SVM-Classifier/Dog-Data/bow.pkl'):
        BoW = kmeans_bow(all_descriptors, num_clusters)
        pickle.dump(BoW, open('C:/Users/Playdata/Desktop/Save-Pets-ML-main/SVM-Classifier/Dog-Data/bow.pkl', 'wb'))
    else:
        BoW = pickle.load(open('C:/Users/Playdata/Desktop/Save-Pets-ML-main/SVM-Classifier/Dog-Data/bow.pkl', 'rb'))

    X_features = create_features_bow(image_descriptors, BoW, num_clusters)

    #Set model
    X_train = [] 
    X_test = []
    Y_train = []
    Y_test = []
    X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.2, random_state=42, shuffle=True)

    #Train SVM
    svm = sklearn.svm.SVC(C = 100, probability=True)
    svm.fit(X_train, Y_train)

    #Train KNN
    knn = KNeighborsClassifier(n_neighbors=10,  weights='distance', leaf_size=200, p=2)
    knn.fit(X_train, Y_train)

    #predict
    #img_test = cv2.imread(opt.dir + '/test/' + opt.test)
    img_test = histo_clahe('C:/Users/Playdata/Desktop/Save-Pets-ML-main/SVM-Classifier/Dog-Data/test/' + opt.test)

    img = [img_test]
    img_sift_feature = extract_sift_features(img)
    img_bow_feature = create_features_bow(img_sift_feature, BoW, num_clusters)

    #---------------------------------
    #-----------evaluation------------
    #---------------------------------

    #predict SVM
    img_predict = svm.predict(img_bow_feature)
    #predict KNN
    img_predict2 = knn.predict(img_bow_feature)

    #prediction probability
    svm_prob = svm.predict_proba(img_bow_feature)[0][img_predict[0]]
    knn_prob = knn.predict_proba(img_bow_feature)[0][img_predict2[0]]

    # print("SVM prob: ", svm.predict_proba(img_bow_feature))
    # print("KNN prob: ", knn.predict_proba(img_bow_feature))
    
    for key, value in label2id.items():
        if value == img_predict[0]:
            svm_k = key
        if value == img_predict2[0]:
            knn_k = key

    result =""
    if svm_k==knn_k:
        result = result+svm_k+","
    else:
        result = result+"a,"
    # result = result+"202151796꿍1234"+","
    # result =result+"등록된강아지"+","

    if (svm_prob < 0.65 and knn_prob < 0.55) or svm_k != knn_k:
        result = result+"미등록강아지"+","
    else:
        result = result+"등록된강아지"+","
    #Accuracy
    if svm.score(X_test,Y_test) > knn.score(X_test,Y_test):
        result = result+str(svm.score(X_test,Y_test))
    else :
        result = result+str(knn.score(X_test,Y_test))
        
    # print("SVM Score: ", svm.score(X_test, Y_test))
    # print("KNN Score: ", knn.score(X_test, Y_test))
    # print("running time: ", round(time.time() - start, 2))

    print(result)

if __name__ == "__main__":
    main()
import graphlearning as gl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.ndimage import gaussian_filter
from sklearn.metrics import accuracy_score
from scipy import sparse
from sklearn.cluster import KMeans
import urllib.request
import scipy.io
import tifffile as tiff
import os, glob
from sklearn.decomposition import PCA as sklearn_pca
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from sklearn.preprocessing import OneHotEncoder
import sys
import cv2 as cv
from scipy.ndimage.morphology import distance_transform_edt
from joblib import dump, load
import utils

def laplace_learning(train_fvecs, train_labels_fl, test_images, sampled_indx,
                     non_local_means, normalize_D):
    '''
        train_fvecs: training feature vectors, (N,d) numpy array
        train_labels_fl: training labels, (N,) numpy array
        test_images: images for prediction, (M,n,n,c) numpy array
        sampled_indx: (K,) integer numpy array, indices sampled in test_images
        non_local_means: bool, use non_local_means features or not
        normalized_D: bool, use normalized weights or not
        
        output:
        pred_labels: (K,n,n) numpy array, record predicted labels of K sampled images
    '''

    n_test = test_images[sampled_indx].shape[0]
    n_pixel = test_images.shape[1] * test_images.shape[2]
    pred_labels = np.zeros((n_test, test_images.shape[1], test_images.shape[2]))
    n_neighbors = 3
    for j in range(n_test):

        if non_local_means:
            t_fevecs = utils.NonLocalMeans(test_images[sampled_indx][j], n_neighbors)
        else:
            t_fevecs = utils.NonLocalMeans(test_images[sampled_indx][j], 0)

        feature_vectors = np.concatenate((train_fvecs, t_fevecs), axis=0)

        J0, D0 = gl.weightmatrix.knnsearch(feature_vectors, 50, method='annoy', similarity='angular')
        I0 = np.arange(J0.shape[0]).reshape(-1, 1) @ np.ones([1, 50])
        n = I0.shape[0]
        k = I0.shape[1]

        D = D0 * D0
        if normalize_D:
            eps = D[:,k-1]/4
            D = D/eps[:,None]
        D = np.exp(-D)

        # Flatten
        I = I0.flatten()
        J = J0.flatten()
        D = D.flatten()

        # Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
        W = sparse.coo_matrix((D, (I, J)), shape=(n, n)).tocsr()
        W = (W + W.transpose()) / 2

        # W = data_preprocess.WeightMatrix(feature_vectors)
        train_ind = np.arange(len(train_fvecs))

        SSL_GL = gl.ssl.laplace(W, class_priors=None)
        laplace_output = SSL_GL._fit(train_ind, train_labels_fl)

        # laplace_output = gl.graph_ssl(W, train_ind, train_labels_fl, algorithm=algorithm, return_vector=True, norm=norm)
        laplace_labels = np.argmax(laplace_output, axis=1)
        laplace_labels_pred = laplace_labels[len(train_labels_fl):].reshape(test_images.shape[1], test_images.shape[2])

        pred_labels[j] = laplace_labels_pred

    return pred_labels

def laplace_learning(data_dic, rep_set_path, data_dic_path, river_names_path,
                     svm_rf_dic, ):

    dic = np.load(train_test_dic, allow_pickle='TRUE').item()
    # dic = np.load('data/train_test_dic.npy', allow_pickle='TRUE').item()

    train_data = dic['train']['image']
    test_data = dic['test']['image']
    train_labels = dic['train']['label']
    test_labels = dic['test']['label']

    #gl_result = np.load('data/gl_result_parallel.npy', allow_pickle='TRUE').item()
    gl_result = np.load(rep_set_path, allow_pickle='TRUE').item()

    picked_train_fvecs = gl_result['picked_train_fvecs']
    picked_train_labels = gl_result['picked_train_labels']

    # data_dic = np.load('data/data_3_6.npy', allow_pickle='TRUE').item()
    # river_names = np.load('data/river_names_3_6.npy')

    data_dic = np.load(data_dic_path, allow_pickle='TRUE').item()
    river_names = np.load(river_names_path)

    SVM_RF_dic = np.load(svm_rf_dic, allow_pickle='TRUE').item()
    apply_rep_set = True
    use_new_data = True
    non_local_means = True

    river_n = 'Ucayali'

    if apply_rep_set and non_local_means:
        train_fvecs = picked_train_fvecs
        train_labels_fl = picked_train_labels
        rep_suffix = '_rep'
    elif not apply_rep_set and non_local_means:
        train_fvecs = SVM_RF_dic['train']['fvecs']
        train_labels_fl = SVM_RF_dic['train']['labels']
        rep_suffix = ''
    else:
        train_fvecs = SVM_RF_dic['train_pixel']['fvecs']
        train_labels_fl = SVM_RF_dic['train_pixel']['labels']
        rep_suffix = '_pixel'

    if use_new_data:
        test_images = data_dic[river_n]['image']
        test_labels = data_dic[river_n]['label']
        filenames = np.array(data_dic[river_n]['filenames'])
        rep_suffix = rep_suffix + river_n
    else:
        test_images = dic['test']['image']
        test_labels = dic['test']['label']
        filenames = dic['test']['filenames']
import graphlearning as gl
import numpy as np
import scipy.sparse as sparse
import utils
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import argparse

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

def data_sampling(train_data, train_labels, num_sample_c=500, n_neighbors=3, randseed=123,
                  include_pixel_features=True, save_path='data/SVM_RF_data.npy'):
    '''
    Sample training feature vectors for SVM and RF
    For each image, sample num_sample_c feature vectors of each class
    Output the sampled data as a dictionary
    Input:
        train_data: N*m*n*C numpy array, m*n*c is the size of each training figure
        train_labels: N*m*n numpy array
        num_sample_c: int, number of pixel sampled for each class
        n_neighbors: int, number of neighbors (patch size is 2*n_neighbors+1) in the non-local means feature extraction process
        randseed: int, random seed
        include_pixel_features: bool, include feature of just pixel values or not
        save_path: str, the path that output will be saved to, empty means don't save
    Output:
        dic_sampled_train: dic, keys: ['train','train_pixel']
    '''

    np.random.seed(randseed)
    N = train_data.shape[0]

    for i in range(N):
      fvecs = utils.NonLocalMeans(train_data[i], n_neighbors)
      gt_labels = train_labels[i].flatten()

      c_names = np.unique(gt_labels)
      C = len(c_names)

      picked_indx = np.array([], dtype=int)
      all_indx = np.arange(gt_labels.shape[0], dtype=int)

      num_append = np.array([])

      for j in range(C):
        index_c = all_indx[gt_labels == c_names[j]]
        n_c = len(index_c)

        if n_c <= num_sample_c:
          picked_indx = np.append(picked_indx, index_c)
          num_append = np.append(num_append, n_c)
        else:
          index_chosen = index_c[np.random.permutation(n_c)][:num_sample_c]
          picked_indx = np.append(picked_indx, index_chosen)
          num_append = np.append(num_append, num_sample_c)

      if i == 0:
        Fvecs = fvecs[picked_indx]
        GT_Labels = gt_labels[picked_indx]
        Num_Append = num_append
      else:
        Fvecs = np.concatenate((Fvecs, fvecs[picked_indx]), axis = 0)
        GT_Labels = np.concatenate((GT_Labels, gt_labels[picked_indx]), axis = 0)
        Num_Append += num_append

    dic_sampled_train = {'train': {'fvecs': Fvecs, 'labels': GT_Labels}}
    if include_pixel_features:
        for i in range(N):
            fvecs = utils.NonLocalMeans(train_data[i], 0)
            gt_labels = train_labels[i].flatten()

            c_names = np.unique(gt_labels)
            C = len(c_names)

            picked_indx = np.array([], dtype=int)
            all_indx = np.arange(gt_labels.shape[0], dtype=int)

            num_append = np.array([])

            for j in range(C):
                index_c = all_indx[gt_labels == c_names[j]]
                n_c = len(index_c)

                if n_c <= num_sample_c:
                    picked_indx = np.append(picked_indx, index_c)
                    num_append = np.append(num_append, n_c)
                else:
                    index_chosen = index_c[np.random.permutation(n_c)][:num_sample_c]
                    picked_indx = np.append(picked_indx, index_chosen)
                    num_append = np.append(num_append, num_sample_c)

            if i == 0:
                Fvecs = fvecs[picked_indx]
                GT_Labels = gt_labels[picked_indx]
                Num_Append = num_append
            else:
                Fvecs = np.concatenate((Fvecs, fvecs[picked_indx]), axis=0)
                GT_Labels = np.concatenate((GT_Labels, gt_labels[picked_indx]), axis=0)
                Num_Append += num_append

    dic_sampled_train['train_pixel'] = {'fvecs': Fvecs, 'labels': GT_Labels}

    if save_path:
        np.save(save_path, dic_sampled_train)

    return dic_sampled_train


def dataset_preparation(method, train_data_path=None, test_data_path=None, apply_rep_set=True,
                        use_outer_data=False, non_local_means=True,
                        river_n='Ucayali', apply_label_change=False):

    '''
        Prepare training and test datasets for different methods and different experiment setting
        Output the corresponding training & test datasets

        Input:
        method: str, method name, {'SVM','RF','GL','DWM'}
        train_data_path: str, path to raw training data of this method,
                        Can be setted as None to use default options
        test_data_path: str, path to raw test data of this method,
                        Can be setted as None to use default options
        apply_rep_set: bool, use representative set in GL method or not
        use_outer_data: bool, use data from other region as the test data or not
        non_local_means: bool, use non-local means feature vectors or not
        river_n: str, river name, only applied when use_outer_data=True
        apply_label_change: bool, apply label change trick or not, change sediment into land

        Output: dic, with the structure {'method', 'train':{'fvecs', 'labels'},
                  'test':{'images', 'labels', 'filenames'}}
    '''

    if test_data_path:
        data_dic = np.load(test_data_path, allow_pickle='TRUE').item()
    else:
        data_dic = np.load('data/data_3_6.npy', allow_pickle='TRUE').item()

    if use_outer_data:
        if test_data_path:
            data_dic = np.load(test_data_path, allow_pickle='TRUE').item()
        else:
            data_dic = np.load('data/data_3_6.npy', allow_pickle='TRUE').item()
        test_images = data_dic[river_n]['image']
        test_labels = data_dic[river_n]['label']
        filenames = np.array(data_dic[river_n]['filenames'])
    else:
        if test_data_path:
            data_dic = np.load(test_data_path, allow_pickle='TRUE').item()
        else:
            data_dic = np.load('data/train_test_dic.npy', allow_pickle='TRUE').item()
        test_images = data_dic['test']['image']
        test_labels = data_dic['test']['label']
        filenames = data_dic['test']['filenames']

    if method == 'SVM' or method == 'RF':
        if train_data_path:
            SVM_RF_dic = np.load(train_data_path, allow_pickle='TRUE').item()
        else:
            SVM_RF_dic = np.load('data/SVM_RF_data.npy', allow_pickle='TRUE').item()

        if non_local_means:
            train_fvecs = SVM_RF_dic['train']['fvecs']
            train_labels_fl = SVM_RF_dic['train']['labels']
        else:
            train_fvecs = SVM_RF_dic['train_pixel']['fvecs']
            train_labels_fl = SVM_RF_dic['train_pixel']['labels']
    elif method == 'GL':
        if apply_rep_set and non_local_means:
            if train_data_path:
                gl_result = np.load(train_data_path, allow_pickle='TRUE').item()
            else:
                gl_result = np.load('data/gl_result_parallel.npy', allow_pickle='TRUE').item()
            train_fvecs = gl_result['picked_train_fvecs']
            train_labels_fl = gl_result['picked_train_labels']
        else:
            if train_data_path:
                SVM_RF_dic = np.load(train_data_path, allow_pickle='TRUE').item()
            else:
                SVM_RF_dic = np.load('data/SVM_RF_data.npy', allow_pickle='TRUE').item()
            if non_local_means:
                train_fvecs = SVM_RF_dic['train']['fvecs']
                train_labels_fl = SVM_RF_dic['train']['labels']
            else:
                train_fvecs = SVM_RF_dic['train_pixel']['fvecs']
                train_labels_fl = SVM_RF_dic['train_pixel']['labels']
    elif method == 'DWM':
        train_fvecs = None
        train_labels_fl = None
    else:
        raise ValueError('method should be chosen as SVM, RF or GL.')

    if apply_label_change:
        change_label = 0  # sediment to land (0) or water (1)
        if train_labels_fl:
            train_labels_fl[train_labels_fl == 2] = change_label
        test_labels[test_labels == 2] = change_label

    output_dic = {'method':method, 'train':{'fvecs':train_fvecs, 'labels':train_labels_fl},
                  'test':{'images':test_images, 'labels':test_labels, 'filenames':filenames}}
    return output_dic

def result_output(method, data_dic, make_plot=True, csv_prefix=None, image_prefix=None,
                  simple_output=False, image_save_folder='output_figures', rgb_normalize=3,
                  apply_label_change=False, DWM_path=None, use_outer_data=False, river_n='Ucayali',
                  apply_sampling=False, downsample_step=3, non_local_means=True):

    '''
        Output results with figures and accuracy values

        Input:
        method: str, method name, chosen from {'SVM','RF','GL','DWM'}
        data_dic: dic, with the same structure as the output of function 'dataset_preparation'
                       when method == 'DWM'
        make_plot: bool, include plots in the output or not
        csv_prefix: str, the prefix of output csv file,
                         None: use default option
        image_prefix: str, the prefix of output image filenames,
                         None: use default option
        simple_output: bool, make the output csv simple or not,
                             simple means to not include results of each image
        image_save_folder: str, the folder where output images are saved to
        rgb_normalize: str, the parameter to normalize the output rgb figure
                            to make it easier to view
        apply_label_change: bool, change the sediment labels into land labels or not,
                            Notice: if method='DWM' and apply_label_change=True, the original DWM will be applied
        DWM_path: str, the path where DeepWaterMap outputs are saved to
                       None: use default option
        use_outer_data: bool, use data from other region as the test data or not
        river_n: str, river name, only applied when use_outer_data=True
        apply_sampling: bool, downsample the test images or not
        dowmsample_step: int, the step size when downsampling

        Output:
        df: pandas table
    '''

    test_images = data_dic['test']['images']
    test_labels = data_dic['test']['labels']
    filenames = data_dic['test']['filenames']
    train_fvecs = data_dic['train']['fvecs']
    train_labels_fl = data_dic['train']['labels']

    n_neighbors = int(((train_fvecs.shape[1] / test_images.shape[3]) ** (1 / 2) - 1 ) / 2)

    if apply_sampling:
        sampled_indx = np.arange(0, test_images.shape[0], downsample_step)
    else:
        sampled_indx = np.arange(0, test_images.shape[0])
    n_test = len(sampled_indx)

    if method != data_dic['method']:
        raise ValueError('method should be the same as the method recorded in the data_dic')

    if not csv_prefix:
        csv_prefix = method + '_results_'
    if not image_prefix:
        image_prefix = method + '_'

    if apply_label_change:
        csv_name = csv_prefix + 'SedToLand' + '.csv'
        image_prefix = image_prefix + 'SedToLand'
    else:
        csv_name = csv_prefix + '.csv'

    if method == 'DWM':
        if DWM_path:
            dwm_dic = np.load(DWM_path, allow_pickle='TRUE').item()
        else:
            dwm_dic = np.load('data/our_DWM_dic.npy', allow_pickle='TRUE').item()

        if apply_label_change and use_outer_data:
            pred_labels = dwm_dic[river_n]['original_pred']
        elif apply_label_change and not use_outer_data:
            pred_labels = dwm_dic['test_data']['original_pred']
        elif not apply_label_change and use_outer_data:
            pred_labels = dwm_dic[river_n]['our_dwm_pred']
        else:
            pred_labels = dwm_dic['test_data']['our_dwm_pred']
        pred_labels = pred_labels[sampled_indx]

    elif method == 'SVM':
        linear_svc = svm.SVC(kernel='linear')
        linear_svc.fit(train_fvecs, train_labels_fl)
        pred_labels = np.zeros_like(test_labels[sampled_indx])
        for j in range(n_test):
            if non_local_means:
                test_fvecs = utils.NonLocalMeans(test_images[sampled_indx][j], n_neighbors)
            else:
                test_fvecs = utils.NonLocalMeans(test_images[sampled_indx][j], 0)
            temp_pred_labels = linear_svc.predict(test_fvecs)
            pred_labels[j] = temp_pred_labels.reshape(test_images.shape[1], test_images.shape[2])

    elif method == 'RF':
        random_forest_clf = RandomForestClassifier(max_depth=50, random_state=15)
        random_forest_clf.fit(train_fvecs, train_labels_fl)
        pred_labels = np.zeros_like(test_labels[sampled_indx])
        for j in range(n_test):
            if non_local_means:
                test_fvecs = utils.NonLocalMeans(test_images[sampled_indx][j], n_neighbors)
            else:
                test_fvecs = utils.NonLocalMeans(test_images[sampled_indx][j], 0)
            temp_pred_labels = random_forest_clf.predict(test_fvecs)
            pred_labels[j] = temp_pred_labels.reshape(test_images.shape[1], test_images.shape[2])

    elif method == 'GL':
        pred_labels = laplace_learning(train_fvecs, train_labels_fl, test_images, sampled_indx,
                         non_local_means, normalize_D=False)

    _, _, _, _, _, df = utils.output_accuracy(test_images[sampled_indx], test_labels[sampled_indx], pred_labels,
                                                  filenames[sampled_indx], make_plot=make_plot, csv_name=csv_name,
                                                  image_prefix=image_prefix, simple_output=simple_output,
                                                  image_save_folder=image_save_folder, rgb_normalize=rgb_normalize)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='GL',
                        help='Method used to classify unlabeled pixels. Method should be GL, DWM, SVM or RF.')
    parser.add_argument('--use_outer_data', type=bool, default=False,
                        help='Use outer data or not. Outer data refers to images in different region from the training set.')
    parser.add_argument('--non_local_means', type=bool, default=True,
                        help='Use non-local means feature extraction or not')
    parser.add_argument('--river_n', type=str, default='Ucayali',
                        help='The river name of the outer dataset')
    parser.add_argument('--apply_label_change', type=bool, default=False,
                        help='Change the sediment labels into land labels or not')
    parser.add_argument('--down_sample', type=bool, default=False,
                        help='Down sample the test dataset to speed up the process or not')
    parser.add_argument('--image_save_folder', type=str, default='output_figures',
                        help='The path where output images are saved to')
    args = parser.parse_args()

    data_dic = dataset_preparation(args.method, use_outer_data=args.use_outer_data,
                                   non_local_means=args.non_local_means, river_n=args.river_n,
                                   apply_label_change=args.apply_label_change)
    df = result_output(args.method, data_dic, image_save_folder=args.image_save_folder,
                       apply_sampling=args.down_sample)
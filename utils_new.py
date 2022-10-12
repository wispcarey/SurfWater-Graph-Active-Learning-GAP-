import graphlearning as gl
import graphlearning.active_learning as al
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import scipy.sparse as sparse
from scipy.ndimage.morphology import distance_transform_edt
import data_preprocess

def local_maxes_k(knn_ind, acq_array, k, top_cut=None, thresh=None):
    # Look at the k nearest neighbors
    # If weights(v) >= weights(u) for all u in neighbors, then v is a local max
    local_maxes = []
    K = knn_ind.shape[1]
    if k > K:
        k = K

    # TODO: acq should be passed in as an array with all the values. Put 0 for labeled set
    n = len(acq_array)
    for i in range(n):
        neighbors = knn_ind[i, :k]  # Indices for neighbors of 1
        acq_vals = acq_array[neighbors]
        if acq_array[i] >= np.max(acq_vals):
            local_maxes.append(i)

    local_maxes = np.array(local_maxes)
    if top_cut:
        acq_max_vals = acq_array[local_maxes]
        local_maxes = local_maxes[np.argsort(acq_max_vals)[-top_cut:]]
    if thresh:
        acq_max_vals = acq_array[local_maxes]
        local_maxes = local_maxes[acq_max_vals > thresh * np.max(acq_max_vals)]

    return local_maxes

def local_maxes_k_new(knn_ind, acq_array, k, top_num, thresh=0):
    # Look at the k nearest neighbors
    # If weights(v) >= weights(u) for all u in neighbors, then v is a local max
    local_maxes = np.array([])
    K = knn_ind.shape[1]
    if k > K:
        k = K

    sorted_ind = np.argsort(acq_array)[::-1]
    local_maxes = np.append(local_maxes, sorted_ind[0])
    global_max_val = acq_array[sorted_ind[0]]
    neighbors = knn_ind[sorted_ind[0], :k]
    sorted_ind = np.setdiff1d(sorted_ind, neighbors, assume_unique=True)

    while len(local_maxes) < top_num and len(sorted_ind) > 0:
        current_max_ind = sorted_ind[0]
        neighbors = knn_ind[current_max_ind, :k]
        acq_vals = acq_array[neighbors]
        sorted_ind = np.setdiff1d(sorted_ind, neighbors, assume_unique=True)
        if acq_array[current_max_ind] >= np.max(acq_vals):
            if acq_array[current_max_ind] < thresh * global_max_val:
                break
            local_maxes = np.append(local_maxes, current_max_ind)

    return local_maxes.astype(int)


def acc_terminal_AL(train_image, train_label, knn_num, n_neighbors, initial_num=10,
                    acq_funs=['mcvopt', 'uc'], terminal_para=None, GL_method='Laplace',
                    local_max_para=(np.inf, 5, 0), acc_metrics=['overall', 'bd3'],
                    class_priors=None):
    if terminal_para:
        epsilon, max_iter = terminal_para
    else:
        epsilon = 1e-3
        max_iter = 1000

    class_names = np.unique(train_label)
    num_classes = len(class_names)

    k, batch_size, thresh = local_max_para

    feature_vectors = data_preprocess.NonLocalMeans(train_image, n_neighbors)
    labels_fl = train_label.flatten()
    knn_data = gl.weightmatrix.knnsearch(feature_vectors, knn_num, method='annoy', similarity='angular')

    # gl package weight matrix
    # W = gl.weightmatrix.knn(feature_vectors, 50, kernel = 'gaussian', knn_data=knn_data)

    # my weight matrix
    J0, D0 = knn_data
    J0, D0 = gl.weightmatrix.knnsearch(feature_vectors, 50, method='annoy', similarity='angular')
    I0 = np.arange(J0.shape[0]).reshape(-1, 1) @ np.ones([1, 50])
    n = I0.shape[0]
    k = I0.shape[1]

    D = D0 * D0
    # eps = D[:,k-1]/4
    # D = D/eps[:,None]
    D = np.exp(-D)

    # Flatten
    I = I0.flatten()
    J = J0.flatten()
    D = D.flatten()

    # Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((D, (I, J)), shape=(n, n)).tocsr()
    W = (W + W.transpose()) / 2

    G = gl.graph(W)
    initial = gl.trainsets.generate(labels_fl, rate=initial_num).tolist()

    if GL_method == 'Laplace':
        model = gl.ssl.laplace(W, class_priors=class_priors)
    elif GL_method == 'RW_Laplace':
        model = gl.ssl.laplace(W, class_priors, reweighting='poisson')
    elif GL_method == 'Poisson':
        model = gl.ssl.poisson(W, class_priors)

    acq_f_list = []
    for acq_fun in acq_funs:
        if acq_fun == 'mc':
            acq_f_list.append(al.model_change())
        elif acq_fun == 'vopt':
            acq_f_list.append(al.v_opt())
        elif acq_fun == 'uc':
            acq_f_list.append(al.uncertainty_sampling())
        elif acq_fun == 'mcvopt':
            acq_f_list.append(al.model_change_vopt())

    act = al.active_learning(W, initial, labels_fl[initial], eval_cutoff=min(200, feature_vectors.shape[0] // 2))
    u = model.fit(act.current_labeled_set, act.current_labels)
    act.candidate_inds = np.setdiff1d(act.training_set, act.current_labeled_set)
    current_label_guesses = model.predict()

    # calculate accuries
    acc_record = np.array([]).astype(np.float64)
    dist_maps = []
    for acc_metric in acc_metrics:
        if acc_metric == 'overall':
            acc_record = np.append(acc_record, np.sum(current_label_guesses == labels_fl) / u.shape[0])
        elif acc_metric[:2] == 'bd':
            d = int(acc_metric[2:])
            dist_map = np.zeros_like(train_label, dtype=np.float64)
            for i in range(num_classes):
                dist_map += distance_transform_edt((train_label == i).astype(np.float64))
            dist_ind = dist_map <= d
            dist_maps.append(dist_ind)
            dist_acc = np.sum(
                current_label_guesses.reshape(train_label.shape[0], train_label.shape[1])[dist_ind] == train_label[
                    dist_ind]) / np.sum(dist_ind)
            acc_record = np.append(acc_record, dist_acc)

    acc_diff = 1

    list_num_labels = [len(act.current_labeled_set)]
    list_acc = [acc_record]

    iter = 0

    while acc_diff > epsilon and iter < max_iter:
        act.candidate_inds = np.setdiff1d(act.training_set, act.current_labeled_set)

        ## active learning
        modded_acq_list = []
        for acq_fun, acq_f in zip(acq_funs, acq_f_list):
            if acq_fun in ['mc', 'uc', 'mcvopt']:
                acq_val = acq_f.compute_values(act, u)
            elif acq_fun == 'vopt':
                acq_val = acq_f.compute_values(act)
            modded_acq_val = np.zeros(feature_vectors.shape[0])
            modded_acq_val[act.candidate_inds] = acq_val
            modded_acq_list.append(modded_acq_val)

        batch_list = []
        for modded_acq_val in modded_acq_list:
            if batch_size > 1:
                batch = local_maxes_k(knn_data[0], modded_acq_val, k, batch_size, thresh)
            else:
                batch = np.argmax(modded_acq_val)
            batch_list.append(batch)
            act.update_labeled_data(batch, labels_fl[batch])

        u = model.fit(act.current_labeled_set, act.current_labels)
        current_label_guesses = model.predict()

        # accuries
        acc_record_new = np.array([]).astype(np.float64)

        dist_map_ind = 0
        for acc_metric in acc_metrics:
            if acc_metric == 'overall':
                acc_record_new = np.append(acc_record_new, np.sum(current_label_guesses == labels_fl) / u.shape[0])
            elif acc_metric[:2] == 'bd':
                dist_ind = dist_maps[dist_map_ind]
                dist_acc = np.sum(
                    current_label_guesses.reshape(train_label.shape[0], train_label.shape[1])[dist_ind] == train_label[
                        dist_ind]) / np.sum(dist_ind)
                acc_record_new = np.append(acc_record_new, dist_acc)
                dist_map_ind += 1

        acc_diff = np.max(np.abs(acc_record_new - acc_record))
        acc_record = acc_record_new

        list_num_labels.append(len(act.current_labeled_set))
        list_acc.append(acc_record)

        # print(len(list_acc))
        # print(len(list_acc[-1]))

        # if not iter%5:
        #   print('Iteration: ' + str(iter) + ', Acc:' + str(acc) + ', Acc diff: ' + str(acc_diff) + 'Current Label Number: ' + str(len(act.current_labeled_set)))

        iter += 1

    labeled_ind = act.current_labeled_set
    act.reset_labeled_data()

    return labeled_ind, list_num_labels, list_acc


def SW_AL(train_images, train_labels, knn_num=50, initial_num=10, acq_funs=['mcvopt', 'uc'], terminal_para=None,
          local_max_para=(np.inf, 10, 0), n_neighbors=3, verbose=True, acc_metrics=['overall', 'bd3'],
          GL_method='Laplace', class_priors=None):
    N = train_images.shape[0]
    for i in range(N):
        train_image = train_images[i]
        train_label = train_labels[i]

        labeled_ind, list_num_labels, list_acc = acc_terminal_AL(train_image, train_label,
                                                                 knn_num, n_neighbors, initial_num,
                                                                 acq_funs, terminal_para, GL_method,
                                                                 local_max_para, acc_metrics,
                                                                 class_priors)

        t_fevecs = data_preprocess.NonLocalMeans(train_image, n_neighbors)
        labels_fl = train_label.flatten()

        if i == 0:
            picked_feature_vecs = t_fevecs[labeled_ind]
            picked_labels = labels_fl[labeled_ind]
        else:
            picked_feature_vecs = np.concatenate((picked_feature_vecs, t_fevecs[labeled_ind]), axis=0)
            picked_labels = np.concatenate((picked_labels, labels_fl[labeled_ind]), axis=0)

        if verbose:
            print('Number of picked feature vectors:', len(labeled_ind))
            acc_info = 'Final accuracy for the current image: '
            for acc_metric, acc_val in zip(acc_metrics, list_acc[-1]):
                acc_info += acc_metric + ': %.2f' % (acc_val * 100)
            print(acc_info)

    return picked_feature_vecs, picked_labels


def gl_ssl(train_fvecs, train_labels_fl, test_images, method, sampled_indx, batch_size=256 ** 2, reweighting='none',
           epsilon=1e-5):
    class_labels = np.unique(train_labels_fl)
    num_class = len(class_labels)

    n_test = test_images[sampled_indx].shape[0]
    n_pixel = test_images.shape[1] * test_images.shape[2]
    pred_labels = np.zeros(test_images.shape[:3])
    pred_confidence = np.zeros(test_images.shape[:3])
    pred_outputs = np.zeros([*test_images.shape[:3], num_class])

    n_neighbors = 3
    for j in range(n_test):
        print('image index: ', j)
        t_fevecs = data_preprocess.NonLocalMeans(test_images[sampled_indx][j], n_neighbors)
        num_batch = np.ceil(t_fevecs.shape[0] / batch_size).astype(int)
        full_labels_pred = np.zeros((t_fevecs.shape[0],))
        full_confidence = np.zeros((t_fevecs.shape[0],))
        full_outputs = np.zeros((t_fevecs.shape[0], num_class))

        for batch_ind in range(num_batch):
            print('batch ind: ', batch_ind)
            batch_fevecs = t_fevecs[(batch_ind * batch_size):((batch_ind + 1) * batch_size)]
            feature_vectors = np.concatenate((train_fvecs, batch_fevecs), axis=0)

            # knn_data = gl.weightmatrix.knnsearch(feature_vectors, 50, method='annoy', similarity='angular')
            # W = gl.weightmatrix.knn(feature_vectors, 50, kernel = 'gaussian', knn_data=knn_data)

            J0, D0 = gl.weightmatrix.knnsearch(feature_vectors, 50, method='annoy', similarity='angular')
            I0 = np.arange(J0.shape[0]).reshape(-1, 1) @ np.ones([1, 50])
            n = I0.shape[0]
            k = I0.shape[1]

            D = D0 * D0
            # eps = D[:,k-1]/4
            # D = D/eps[:,None]
            D = np.exp(-D)

            # Flatten
            I = I0.flatten()
            J = J0.flatten()
            D = D.flatten()

            # Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
            W = sparse.coo_matrix((D, (I, J)), shape=(n, n)).tocsr()
            W = (W + W.transpose()) / 2

            ### reweighting
            if epsilon > 0:
                G = gl.graph(W)
                n = W.shape[0]
                f = np.zeros(n)
                f[:len(train_fvecs)] = 1
                f -= np.mean(f)
                L = G.laplacian() + sparse.spdiags(epsilon * np.ones(n), 0, n, n).tocsr()
                w = gl.utils.conjgrad(L, f, tol=1e-5)
                w -= np.min(w)
                w += 1e-5
                D = sparse.spdiags(w, 0, n, n).tocsr()
                W = D * W * D

            train_ind = np.arange(len(train_fvecs))

            if method == 'Laplace':
                SSL_GL = gl.ssl.laplace(W, class_priors=None, reweighting=reweighting)
            elif method == 'Poisson':
                SSL_GL = gl.ssl.poisson(W, class_priors=None, reweighting=reweighting)
            elif method == 'Multiclass_mbo':
                SSL_GL = gl.ssl.multiclass_mbo(W, class_priors=None, reweighting=reweighting)
            elif method == 'Modularity_mbo':
                SSL_GL = gl.ssl.modularity_mbo(W, class_priors=None, reweighting=reweighting)
            elif method == 'Randomwalk':
                SSL_GL = gl.ssl.randomwalk(W, class_priors=None, reweighting=reweighting)

            gl_output = SSL_GL._fit(train_ind, train_labels_fl)

            gl_labels = np.argmax(gl_output, axis=1)
            gl_confidence = np.var(gl_output, axis=1)
            full_labels_pred[(batch_ind * batch_size):((batch_ind + 1) * batch_size)] = gl_labels[len(train_labels_fl):]
            full_confidence[(batch_ind * batch_size):((batch_ind + 1) * batch_size)] = gl_confidence[
                                                                                       len(train_labels_fl):]
            full_outputs[(batch_ind * batch_size):((batch_ind + 1) * batch_size), :] = gl_output[len(train_labels_fl):,
                                                                                       :]

        gl_confidence_pred = full_confidence.reshape(test_images.shape[1], test_images.shape[2])
        gl_labels_pred = full_labels_pred.reshape(test_images.shape[1], test_images.shape[2])
        gl_outputs_pred = full_outputs.reshape(test_images.shape[1], test_images.shape[2], num_class)

        pred_labels[j] = gl_labels_pred
        pred_confidence[j] = gl_confidence_pred
        pred_outputs[j] = gl_outputs_pred

    return pred_labels, pred_outputs, pred_confidence

import pandas as pd
def output_accuracy(img_data, gt_labels, pred_labels, filenames,
                    make_plot=False, csv_name='results.csv', save_csv=True,
                    image_prefix='sample_figs', simple_output=False, save_img=False,
                    image_save_folder='sample_images', rgb_normalize=None, dist_val=(3,10)):
    if len(dist_val) != 2:
        raise ValueError('dist_val should be a tuple with length 2')
    d1, d2 = dist_val

    class_names = np.unique(gt_labels)
    num_class = len(class_names)
    num_pixels = gt_labels.shape[1] * gt_labels.shape[2]

    class_correct_num = np.zeros([num_class, gt_labels.shape[0]])
    class_gt_num = np.zeros([num_class, gt_labels.shape[0]])
    class_FP_num = np.zeros([num_class, gt_labels.shape[0]])
    accuracy = np.zeros(gt_labels.shape[0])

    dist_1 = np.zeros([3, gt_labels.shape[0]])
    dist_2 = np.zeros([3, gt_labels.shape[0]])

    for image_ind in range(gt_labels.shape[0]):

        test_labels_pred = pred_labels[image_ind]
        test_labels_gt = gt_labels[image_ind]

        dist_map = np.zeros_like(test_labels_gt, dtype=np.float64)
        for i in range(num_class):
            ind = test_labels_gt == i
            pred_ind = test_labels_pred == i
            class_correct_num[i, image_ind] = np.sum(test_labels_pred[ind] == test_labels_gt[ind])
            class_FP_num[i, image_ind] = np.sum(test_labels_pred[pred_ind] != test_labels_gt[pred_ind])
            class_gt_num[i, image_ind] = np.sum(ind)
            dist_map += distance_transform_edt((test_labels_gt == i).astype(int))

        dist_1[0, image_ind] = np.sum(test_labels_pred[dist_map <= d1] == test_labels_gt[dist_map <= d1])
        dist_1[1, image_ind] = np.sum(dist_map <= d1)
        dist_1[2, image_ind] = dist_1[0, image_ind] / dist_1[1, image_ind]

        dist_2[0, image_ind] = np.sum(test_labels_pred[dist_map <= d2] == test_labels_gt[dist_map <= d2])
        dist_2[1, image_ind] = np.sum(dist_map <= d2)
        dist_2[2, image_ind] = dist_2[0, image_ind] / dist_2[1, image_ind]

        accuracy[image_ind] = np.sum(class_correct_num[:, image_ind]) / np.sum(class_gt_num[:, image_ind])

        if make_plot:
            (filepath, tempfilename) = os.path.split(filenames[image_ind])
            (filename, _) = os.path.splitext(tempfilename)

            img = img_data[image_ind]
            if rgb_normalize is None:
                modified_img = img[:, :, [2, 1, 0]] / np.max(img[:, :, [2, 1, 0]])
            else:
                modified_img = img[:, :, [2, 1, 0]] * rgb_normalize
                if np.max(modified_img > 1):
                    modified_img = modified_img / np.max(modified_img)
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
            ax0.imshow(modified_img)
            ax1.imshow(test_labels_pred)
            ax2.imshow(test_labels_gt)
            if save_img:
                plt.imsave(os.path.join(image_save_folder, filename[:-7] + 'rgb.png'), modified_img)
            if num_class == 2:
                save_img_pred = test_labels_pred
                save_img_gt = test_labels_gt
                save_img_pred[0, 0] = 2
                save_img_gt[0, 0] = 2
                if save_img:
                    plt.imsave(os.path.join(image_save_folder, image_prefix + filename[:-7] + '_pred.png'),
                              save_img_pred)
                    plt.imsave(os.path.join(image_save_folder, filename[:-7] + 'gt2.png'), save_img_gt)
            else:
                if save_img:
                    plt.imsave(os.path.join(image_save_folder, image_prefix + filename[:-7] + '_pred.png'),
                              test_labels_pred)
                    plt.imsave(os.path.join(image_save_folder, filename[:-7] + 'gt.png'), test_labels_gt)
            plt.show()

    total_acc = np.sum(class_correct_num) / np.sum(class_gt_num)
    class_total_acc = np.sum(class_correct_num, axis=1) / np.sum(class_gt_num, axis=1)
    class_acc = class_correct_num / class_gt_num
    class_FPR = class_FP_num / (num_pixels - class_gt_num)
    class_total_FPR = np.sum(class_FP_num, axis=1) / np.sum(num_pixels - class_gt_num, axis=1)

    class_num_ratio = class_gt_num / num_pixels
    nor_class_FPR = class_FPR / class_num_ratio
    class_total_NFPR = class_total_FPR / np.sum(class_gt_num, axis=1) * np.sum(class_gt_num)

    sum_dist_1 = np.sum(dist_1, axis=1)
    sum_dist_2 = np.sum(dist_2, axis=1)

    if num_class == 2:
        df = pd.DataFrame.from_dict({"Land TPR": [f"{num * 100:.2f}" for num in class_acc[0, :]],
                                     "Land FPR": [f"{num * 100:.2f}" for num in class_FPR[0, :]],
                                     "Land NFPR": [f"{num * 100:.2f}" for num in nor_class_FPR[0, :]],
                                     "Water TPR": [f"{num * 100:.2f}" for num in class_acc[1, :]],
                                     "Water FPR": [f"{num * 100:.2f}" for num in class_FPR[1, :]],
                                     "Water NFPR": [f"{num * 100:.2f}" for num in nor_class_FPR[1, :]],
                                     ("Dist" + str(d1)): [f"{num * 100:.2f}" for num in dist_1[2, :]],
                                     ("Dist" + str(d2)): [f"{num * 100:.2f}" for num in dist_2[2, :]],
                                     "Whole": [f"{num * 100:.2f}" for num in accuracy]}, orient='index',
                                     columns=[i for i in range(class_acc.shape[1])])
        df['Total'] = [f"{num * 100:.2f}" for num in [class_total_acc[0], class_total_FPR[0], class_total_NFPR[0],
                                                      class_total_acc[1], class_total_FPR[1], class_total_NFPR[1],
                                                      sum_dist_1[0] / sum_dist_1[1],
                                                      sum_dist_2[0] / sum_dist_2[1], total_acc]]
    elif num_class == 3:
        df = pd.DataFrame.from_dict({"Land TPR": [f"{num * 100:.2f}" for num in class_acc[0, :]],
                                     "Land FPR": [f"{num * 100:.2f}" for num in class_FPR[0, :]],
                                     "Land NFPR": [f"{num * 100:.2f}" for num in nor_class_FPR[0, :]],
                                     "Water TPR": [f"{num * 100:.2f}" for num in class_acc[1, :]],
                                     "Water FPR": [f"{num * 100:.2f}" for num in class_FPR[1, :]],
                                     "Water NFPR": [f"{num * 100:.2f}" for num in nor_class_FPR[1, :]],
                                     "Sediment TPR": [f"{num * 100:.2f}" for num in class_acc[2, :]],
                                     "Sediment FPR": [f"{num * 100:.2f}" for num in class_FPR[2, :]],
                                     "Sediment NFPR": [f"{num * 100:.2f}" for num in nor_class_FPR[2, :]],
                                     ("Dist" + str(d1)): [f"{num * 100:.2f}" for num in dist_1[2, :]],
                                     ("Dist" + str(d2)): [f"{num * 100:.2f}" for num in dist_2[2, :]],
                                     "Whole": [f"{num * 100:.2f}" for num in accuracy]}, orient='index',
                                     columns=[i for i in range(class_acc.shape[1])])
        df['Total'] = [f"{num * 100:.2f}" for num in [class_total_acc[0], class_total_FPR[0], class_total_NFPR[0],
                                                      class_total_acc[1], class_total_FPR[1], class_total_NFPR[1],
                                                      class_total_acc[2], class_total_FPR[2], class_total_NFPR[2],
                                                      sum_dist_1[0] / sum_dist_1[1],
                                                      sum_dist_2[0] / sum_dist_2[1], total_acc]]

    if simple_output:
        df = df.iloc[:, -1]

    if save_csv:
        df.to_csv(csv_name)

    return class_acc, class_FPR, dist_1, dist_2, accuracy, df
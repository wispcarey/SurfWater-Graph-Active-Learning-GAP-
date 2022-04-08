import numpy as np
import sys
import os
import cv2 as cv
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import my_cnn_sw
import argparse

sys.path.insert(1, 'deepwatermap_original')
import inference

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

def our_dwm_results(train_test_path, data_dic_path, river_names_path, checkpoint_path, save_path):

    data_dic = np.load(data_dic_path, allow_pickle='TRUE').item()
    river_names = np.load(river_names_path)
    train_test_dic = np.load(train_test_path, allow_pickle='TRUE').item()

    # load our DWM
    num_class = 3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')
    model = my_cnn_sw.WaterBoundaryCNN(input_num_c=6, num_class=num_class).to(device)

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # output record
    our_dwm_dic = {}

    for j in range(len(river_names)):
        print(river_names[j])
        data_s = torch.from_numpy(data_dic[river_names[j]]['image']).float().permute(0, 3, 1, 2)
        label_s = torch.from_numpy(data_dic[river_names[j]]['label']).float()
        surfwater = MyDataset(data_s, label_s)

        pred_labels = np.zeros_like(data_dic[river_names[j]]['label'])

        model.eval()
        with torch.no_grad():
            for test_ind in range(len(surfwater)):
                data, _ = surfwater[test_ind]
                data = data.to(device)
                data = data.unsqueeze(0)

                x_pred = model(data)
                pred = x_pred.argmax(dim=1).cpu().numpy()[0].astype(int)
                pred_labels[test_ind] = pred

        our_dwm_dic[river_names[j]] = {}
        our_dwm_dic[river_names[j]]['our_dwm_pred'] = pred_labels

    data_tt = torch.from_numpy(train_test_dic['test']['image']).float().permute(0, 3, 1, 2)
    label_tt = torch.from_numpy(train_test_dic['test']['label']).float()
    surfwater_tt = MyDataset(data_tt, label_tt)

    pred_labels_tt = np.zeros_like(train_test_dic['test']['label'])

    model.eval()
    with torch.no_grad():
        for test_ind in range(len(surfwater_tt)):
            data, _ = surfwater_tt[test_ind]
            data = data.to(device)
            data = data.unsqueeze(0)

            x_pred = model(data)
            pred = x_pred.argmax(dim=1).cpu().numpy()[0].astype(int)
            pred_labels_tt[test_ind] = pred

    our_dwm_dic['test_data'] = {}
    our_dwm_dic['test_data']['our_dwm_pred'] = pred_labels

    np.save(save_path, our_dwm_dic)

def original_dwm_results(train_test_path, data_dic_path, river_names_path,
                         save_path, dwm_output_path, suffix='landsat'):

    data_dic = np.load(data_dic_path, allow_pickle='TRUE').item()
    river_names = np.load(river_names_path)

    our_dwm_dic = np.load(save_path, allow_pickle='TRUE').item()
    N = len(suffix)

    for i in range(len(river_names)):
        image_paths = data_dic[river_names[i]]['filenames']
        print(river_names[i])
        pred_labels = np.zeros_like(data_dic[river_names[i]]['label'])
        for j in range(len(image_paths)):
            (filepath, tempfilename) = os.path.split(image_paths[j])
            (filename, extension) = os.path.splitext(tempfilename)
            image_path = os.path.join(dwm_output_path, filename[:-N] + "DWM_original_output.png")

            dwm_img = cv.imread(image_path)
            pred = cv.threshold(dwm_img, 127, 1, cv.THRESH_BINARY)[1][:, :, 0]
            pred_labels[j] = pred

        our_dwm_dic[river_names[i]]['original_pred'] = pred_labels

    train_test_dic = np.load(train_test_path, allow_pickle='TRUE').item()

    image_paths = train_test_dic['test']['filenames']
    pred_labels = np.zeros_like(train_test_dic['test']['label'])
    for j in range(len(image_paths)):
        (filepath, tempfilename) = os.path.split(image_paths[j])
        (filename, extension) = os.path.splitext(tempfilename)
        image_path = os.path.join(dwm_output_path, filename[:-N] + "DWM_original_output.png")

        dwm_img = cv.imread(image_path)
        pred = cv.threshold(dwm_img, 127, 1, cv.THRESH_BINARY)[1][:, :, 0]
        pred_labels[j] = pred

    our_dwm_dic['test_data']['original_pred'] = pred_labels

    np.save(save_path, our_dwm_dic)


def original_dwm_outputs(data_dic_path, river_names_path,
                         checkpoint_path, dwm_output_path, suffix='landsat'):

    data_dic = np.load(data_dic_path, allow_pickle='TRUE').item()
    river_names = np.load(river_names_path)
    N = len(suffix)

    for i in range(len(river_names)):
        image_path = data_dic[river_names[i]]['filenames']
        print(river_names[i])
        for j in range(len(image_path)):
            print(j)
            (filepath, tempfilename) = os.path.split(image_path[j])
            (filename, extension) = os.path.splitext(tempfilename)
            output_path = os.path.join(dwm_output_path, filename[:-N] + "DWM_original_output.png")

            inference.main(checkpoint_path, image_path[j], output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply_retrained_dwm', type=bool, default=True,
                        help='apply the retrained deepwatermap or not (not means to apply the original one)')
    parser.add_argument('--data_dic', type=str, default='data/data_3_6.npy',
                        help='path to the raw data dictionary')
    parser.add_argument('--river_names', type=str, default='data/river_names_3_6.npy',
                        help='path to the list of river names')
    parser.add_argument('--train_test_dic', type=str, default='data/train_test_dic.npy',
                        help='path to the train-test data dictionary')
    parser.add_argument('--checkpoint_path_ours', type=str, default='data/my_deepwatermap.pt',
                        help='path to the checkpoint of our retrained dwm')
    parser.add_argument('--checkpoint_path_original', type=str, default='data/cp.135.ckpt',
                        help='path to the checkpoint of the original dwm')
    parser.add_argument('--save_path', type=str, default='data/our_dwm_dic.npy',
                        help='path to the directory of the train-test data')
    parser.add_argument('--output_path_original', type=str, default='DWM_original_output',
                        help='path to the directory of the output figures of the original dwm')
    args = parser.parse_args()

    # get results of our retrained dwm (3 classes)
    our_dwm_results(args.train_test_dic, args.data_dic, args.river_names,
                    args.checkpoint_path_ours, args.save_path)

    # get outputs of the original dwm
    original_dwm_outputs(args.data_dic, args.river_names,
                         args.checkpoint_path_original, args.output_path_original)

    # get results of our retrained dwm (2 classes)
    original_dwm_results(args.train_test_dic, args.data_dic, args.river_names,
                         args.save_path, args.output_path_original)

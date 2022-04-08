import numpy as np
import argparse
import tifffile as tiff
import os, glob

## each labeled image should be saved in the form as: river_name + anything + suffix + '.tif'
## the corresponding label is saved in the name: river_name + anything + 'labeled.tif'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_img_dir', type=str, default='Labeled_patches 3-6-22',
                        help='path to the directory of the raw data')
    parser.add_argument('--suffix', type=str, default='landsat',
                        help='suffix of the original image')
    parser.add_argument('--data_save_path', type=str, default='data/data_3_6.npy',
                        help='path that the processed data dictionary is saved to')
    parser.add_argument('--rivername_save_path', type=str, default='data/river_names_3_6.npy',
                        help='path that the list of river names is saved to')
    args = parser.parse_args()

    original_img_dir = args.original_img_dir

    path = os.path.join(original_img_dir, '*' + args.suffix + '.tif')
    filenames = glob.glob(path)

    namelist = []

    for file in filenames:
        (_, tempfilename) = os.path.split(file)
        (filename, _) = os.path.splitext(tempfilename)

        namelist.append(filename.split('_')[0])

    river_names = list(set(namelist))
    data_dic = {}

    for rivername in river_names:
        img_path = os.path.join(original_img_dir, rivername + '*' + args.suffix + '.tif')
        img_filenames = glob.glob(img_path)

        for i in range(len(img_filenames)):
            img_filename = img_filenames[i]
            (_, tempfilename) = os.path.split(img_filename)
            (filename, _) = os.path.splitext(tempfilename)
            label_filename = os.path.join(original_img_dir, filename[:-len(args.suffix)] + 'labeled.tif')
            if i == 0:
                label = np.expand_dims(tiff.imread(label_filename), axis=0)
                img = np.expand_dims(tiff.imread(img_filename), axis=0)
            else:
                label = np.append(label, np.expand_dims(tiff.imread(label_filename), axis=0), axis=0)
                img = np.append(img, np.expand_dims(tiff.imread(img_filename), axis=0), axis=0)

        dic = {'label': label, 'image': img, 'filenames': img_filenames}
        data_dic[rivername] = dic

    np.save(args.data_save_path, data_dic)
    np.save(args.rivername_save_path, river_names)
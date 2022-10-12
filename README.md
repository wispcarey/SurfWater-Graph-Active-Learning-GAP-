# SurfWater-Graph-Active-Learning-GAP-
This is a novel graph active learning method to detect surface water and sediment pixel in multispectral satellite images. We developed a graph active learning pipeline, called GAP to do this task.

Some codes here are copied from or modified based on [Kevin Miller's Model-Change Active Learning repository](https://github.com/millerk22/model-change-paper "Model Change Paper") and [Deep Water Map](https://github.com/isikdogan/deepwatermap "Deep Water Map source codes"). Besides, we applied some functions in [Jeff Calder's graph learning package](https://github.com/jwcalder/GraphLearning "Graph Learning Package").

Our experiment are based on the [RiverPIXELS dataset](https://data.ess-dive.lbl.gov/view/doi:10.15485/1865732 "RiverPIXELS") created by Jon Schwenk in Los Alamos National Laboratory (LANL). 

## File Introduction
| File        | Introduction       |  
| ------------- |-------------| 
| `data_process.py`  | Process raw images into our required data structure  |  
| `my_cnn_sw.py`        | The CNN structure (the same as DWM)| 
| `trainer.py`  | Functions to retrain DeepWaterMap  |  
| `DeepWaterMap_ours.py` | Output results of both our retrained DWM and the original DWM | 
| `utils.py`  | Useful functions, including non-local means feature extraction and results output functions |  
| `utils_new.py`  | Some updated utils function, including the accuracy based active learning function | 
| `methods.py`  | Implement different methods, including SVM, RF, DWM and our GAP|  
| `quick_start.py`  | Quick start with your own images|  

We write the original DeepWaterMap codes into *PyTorch* in order to retrain it on our own dataset.

## Dataset Source
In order to reimplement our algorithm, you should process the raw image dataset into some specific format. The processed data can be downloaded from [my google drive (SurfwaterGAP)](https://drive.google.com/drive/folders/17wxkCVneJrozsX-q-9XmyhF09LCfvaNO?usp=sharing "SurfwaterGAP"). You can check the following table to see what's inside each folder in this shared drive. **Notice:** You should download all files in the `data` folder if you want to run our codes directly.
| Folder        | Inside        |  
| ------------- |-------------| 
| Labeled_patches 3-6-22  | Original image dataset (RiverPIXELS)| 
| data          | Processed data with required format| 
| DWM_original_output     | Output labeled images from the original DeepWaterMap  |  
| output_figures | Output figures of different methods  | 

If you want to use your own dataset, you should use `data_process.py` to process your raw data into the required format. In addition, you can use the function *train_test_split* in `utils.py`. You can use functions *our_dwm_results* and *original_dwm_outputs* in `DeepWaterMap_ours.py` to get the outputs of DeepWaterMap about your dataset. For accuracy analysis, in `methods.py`, you should use the function *dataset_preparation* to prepare the required data formation and the function *result_output* to get the output accuracies.

## Reimplement our results & Quick Start with your own images
**Notice:** You should download all files from the data folder to commends in this part.

You can use `methods.py` to reimplement our output result of different methods. Here you can choose the method as the support vector machine (SVM), the random forest (RF), the retrained DeepWaterMap (DWM) and our graph active learning pipeline (GL). As an example, if you want to reimplement the results of SVM:
```python
python methods.py --method SVM 
```
In addition, if you want to have a look at the results of the original DeepWaterMap:
```python
python methods.py --method DWM --apply_label_change True 
```

If you want to have a quick test on your own dataset, you should put your `tif` images in a folder (eg. input_images) and run the following commend
```python
python quick_start.py --image_path input_images --output_path output_images
```
Your tif images must be the size $256\times256\times6$. The 6 channels of the image should sequentially be Blue, Green, Red, Near IR, Shortwave IR 1, Shortwave
IR 2.


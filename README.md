# SurfWater-Graph-Active-Learning-GAP-
This is a novel graph active learning method to detect surface water and sediment pixel in multispectral satellite images. We developed a graph active learning pipeline, called GAP to do this task.

Some codes here are copied from or modified based on [Kevin Miller's Model-Change Active Learning repository](https://github.com/millerk22/model-change-paper "Model Change Paper") and [Deep Water Map](https://github.com/isikdogan/deepwatermap "Deep Water Map source codes"). Besides, we applied some functions in [Jeff Calder's graph learning package](https://github.com/jwcalder/GraphLearning "Graph Learning Package").

Our experiment are based on the [RiverPIXELS dataset](https://data.ess-dive.lbl.gov/view/doi:10.15485/1865732 "RiverPIXELS") created by Jon Schwenk in Los Alamos National Laboratory (LANL). 

## Dataset Source
In order to reimplement our algorithm, you should process the raw image dataset into some specific format. The processed data can be downloaded from [my google drive (SurfwaterGAP)](https://drive.google.com/drive/folders/17wxkCVneJrozsX-q-9XmyhF09LCfvaNO?usp=sharing "SurfwaterGAP"). You can check the following table to see what's inside each folder in this shared drive. **Notice:** You should download all files in the `data` folder if you want to run our codes directly.
| Folder        | Inside        |  
| ------------- |:-------------:| 
| Labeled_patches 3-6-22  | Original image dataset (RiverPIXELS)| 
| data          | processed data with required format| 
| col 2 is      | centered      |  
| zebra stripes | are neat      | 

If you want to use your own dataset, you should use `data_process.py` to process your raw data into the required format.

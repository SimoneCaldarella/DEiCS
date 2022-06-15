# Depth Estimation in Crowd Simulation (DEiSC)

This is the repo for the Computer Vision course project.

The report of the project can be downloaded [here](./cv_report_caldarella_pedeni_2022.pdf).

Click [here](https://drive.google.com/drive/folders/1QeC6BGdSkCF9nwqj8naZApgODzFVYEKU?usp=sharing) to get the access to the converted and resized dataset.

---

### Authors
- Simone Caldarella (simone.caldarella@studenti.unitn.it)
- Federico Pedeni (federico.pedeni@studenti.unitn.it)
    
### Requirements:
- Monodepth2 library --> clone the library in your project directory with ```git clone https://github.com/nianticlabs/monodepth2.git```
- Numpy --> ```pip install numpy```
- Matplotlib --> ```pip install matplotlib```
- PyTorch --> Check [here](https://pytorch.org/get-started/locally/) the command needed to install the version of pytorch that best suits your pc requirements

### Dataset Structure:
/ConvertedDataset/:
- 1:
  - frame_NUM1.png
  - frame_NUM1_depth.png
  - ...
  - frame_NUMN.png
  - frame_NUMN_depth.png
- 2:
  - frame_NUM1.png
  - frame_NUM1_depth.png
  - ...
  - frame_NUMN.png
  - frame_NUMN_depth.png
- 3
- 4
- 5
- 6
- 7

! The structure is the same in all the numbered directories (each one contains a set of pair of images taken from the same camera)
!! NUM (NUM1, NUM2, ..., NUMN) is a 4 digit number -- Images from different cameras can have the same NUM in the name
    
### Usage:
0) Download the dataset from the google drive or use your own dataset:
  - if you use your own dataset make sure to have the correct structure
  - if your dataset is not of the correct size 640x192 png/jpg images run the code with the flag ```__CONVERT_MODE__=True```
1) Update all the global variables (path and flag) according to the paths you want to use (free choice)
2) Run the script



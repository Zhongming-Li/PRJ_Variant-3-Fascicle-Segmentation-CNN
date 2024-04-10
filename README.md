# PRJ: Muscle Analysis in B-mode Ultrasound Images

A software application designed for the automated segmentation and prediction of muscle fascicles and aponeuroses in B-mode ultrasound images of muscles.

## Installation
1. clone this repository. Note that if you are downloading zip, make sure to check the file sizes of the following files：
    - all .h5 files under the directory "models" (larger than 50MB).
    - dist/MuscleAnalysis.exe (422MB).
    - build/MuscleAnalysis/MuscleAnalysis.pkg (422MB).
   If any of these files is only a few KB large, download the files by clicking the "Download raw file" button as shown below and put them in the exact same directories
![0c22412a6876352dd2888b0fd135fb1](https://github.com/Zhongming-Li/PRJ_Variant-3-Fascicle-Segmentation-CNN/assets/114877324/d4dc4de5-0f63-4b77-b2c0-3c77b7108a1a)

2. create and activate a new virtual environment
3. run the following command:
```sh
conda install -c anaconda cudatoolkit==9.0
```
3. run the following command:
```sh
conda install -c anaconda cudnn==7.6.5
```
4. run the following command:
```sh
pip install -r requirements.txt
```


## Usage
#### Software
Run the MuscleAnalysis.exe in the **dist** directory
Make sure you download both the **build** and **dist** folders and put them under the same directory.

Alternatively, get the executable file from: https://drive.google.com/drive/folders/1ZVEWqXA3MNwNkfOXwu7zxDZFTEVnsUHk?usp=drive_link
Make sure you download both the **build** and **dist** folders and put them under the same directory.

To use the graphical user interface (GUI) in python environment, run the following command:
```sh
python MuscleAnalysis.py
```

#### Inference
To use the trained model to analyse a single image, run the following command:
```sh
python inference_image.py --image_path path/to/your/image.png --flip 1 --mm 10 --pixel 50 --save_dir ./test_inference_output
```
arg: 
- **image_path**: path of the image to be analyzed
- **flip**: set to 0 if the fascicles are oriented from bottom-left to top-right, otherwise set to 1.
- **mm**: indicates the conversion ratio between millimeters(mm) and pixels. in this case, 10mm = 50pixels
- **pixel**: indicates the conversion ratio between millimeters(mm) and pixels. in this case, 10mm = 50pixels
- **save_dir**: the directory in which the analyzed image will be saved
- **fasc_model_path**(optional): the path to the pre-trained fascicle segmentation model in HDF5 (.h5) format (e.g. models\model-fasc-WW-unet.h5). A default model is provided if no model is specified.
- **apo_model_path**(optional): the path to the pre-trained aponeurosis segmentation model in HDF5 (.h5) format (e.g. models\model-apo2-nc.h5). A default model is provided if no model is specified.


#### Training
To train your own model, run the following command:
```sh
python train_segmentation.py --image_dir path/to/your/training/images --mask_dir path/to/your/training/masks --seg_type fascicle --model_type attention-u-net --model_name my_model
```
arg:
- **image_dir**: The directory containing original images for the training dataset. Please ensure that the images are directly located within this directory and not stored in subdirectories.
- **mask_dir**: The directory containing binary masks for the training dataset. Please ensure that the images are directly located within this directory and not stored in subdirectories. Note that the mask images must have the exact same name as the corresponding original image.
- **seg_type**: Specify the segmentation task: "aponeurosis" or "fascicle".
- **model_type**(optional): Specify the architecture of the neural network: 'u-net' or 'attention-u-net'. It is set to 'u-net' by default.
- **model_name**(optional): The model name for saving after training. Please provide the name without the ".h5" extension. It is set to "model-seg" by default. The trained model will be saved in thew current directory.



## References
- **Repository:** [DL_Track](https://github.com/njcronin/DL_Track.git)
- **Paper:** [Fully automated analysis of muscle architecture from B-mode ultrasound images with deep learning](https://arxiv.org/abs/2009.04790)



- Repository: [DL Track](https://github.com/njcronin/DL_Track.git)
  - Paper: [Fully automated analysis of muscle architecture from B-mode ultrasound images with deep learning](https://arxiv.org/abs/2009.04790)




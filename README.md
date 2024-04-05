# PRJ: Muscle Analysis in B-mode Ultrasound Images

A software application designed for the automated segmentation and prediction of muscle fascicles and aponeuroses in B-mode ultrasound images of muscles.

## Installation

To set up the environment, run the following command:

```sh
pip install -r requirements.txt
```


## Usage
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

- Repository: [DL Track](https://github.com/njcronin/DL_Track.git)
  - Paper: [Fully automated analysis of muscle architecture from B-mode ultrasound images with deep learning](https://arxiv.org/abs/2009.04790)




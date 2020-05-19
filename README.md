
# Toward a Universal Model for Shape from Texture

This repository contains code for:

**[Toward a Universal Model for Shape from Texture](http://vision.seas.harvard.edu/sft/)**
[Dor Verbin](https://scholar.harvard.edu/dorverbin) and [Todd Zickler](http://www.eecs.harvard.edu/~zickler/)
CVPR 2020


Please contact us by email for questions about our paper or code.




## Requirements

Our code is implemented in tensorflow. It has been tested using tensorflow 1.9 but it should work for other tensorflow 1.x versions. The following packages are required:

- python 3.x
- tensorflow 1.x
- numpy >= 1.14.0
- pillow >= 5.1.0
- matplotlib >= 2.2.2 (only used for plotting during training)



## Running the model

To run our model execute the following:
```
python train.py -image_path <path to input image> -output_folder <output folder>
```

In order to plot results during training using `matplotlib`, specify `-do_plot True`. To save the models, use `-do_save_model True`.

Using an NVIDIA Tesla V100 GPU, training takes about 110 minutes for a `640x640` image.




## Data

(Coming soon)



## Citation

For citing our paper, please use:
```
@InProceedings{verbin2020sft,
title={Toward a Universal Model for Shape from Texture},
author = {Verbin, Dor and Zickler, Todd},
booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2020}
}
```

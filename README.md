
# Toward a Universal Model for Shape from Texture

This repository contains code for:

**[Toward a Universal Model for Shape from Texture](http://vision.seas.harvard.edu/sft/)**
<br>
[Dor Verbin](https://scholar.harvard.edu/dorverbin) and [Todd Zickler](http://www.eecs.harvard.edu/~zickler/)
<br>
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

Our synethetic dataset was generated in [Blender](http://www.blender.org) by using cloth simulation. The texture images were mapped onto a square mesh
and dropped onto a surface. After the simulation is done running, the result is rendered. Blender also enables extracting the ground truth surface normals
by saving them into an `.stl` file (go to `Export > Stl` and make sure "Selection Only" is checked).

We provide two files below:
- A zip file containing all images can be downloaded [here](http://vision.seas.harvard.edu/sft/data/images.zip) (15.2 MB).
- A zip file containing all source Blender files and `.stl` files can be downloaded [here](http://vision.seas.harvard.edu/sft/data/models.zip) (22.8 MB).

The file containing all images contains five directories corresponding to the four shapes in the paper plus one containing the original (flat) images.
The file containing the Blender models has four directories corresponding to the four shapes. Each one contains a Blender
file with an embedded python script which can be run to automatically render all images used in the paper. Each directory also contains an `.stl` file extracted from Blender, which stores the
true shape. The script can also be used to generate the images from the paper, with shading and specular highlights.
The `sphere` directory also contains the Blender files and embedded python scripts used to generate the images for Figures 7 and 8 from our paper.
Note: In order to use the Blender files, the two zip files must be unzipped in the same directory (only the flat directory is used by the Blender files).



## Citation

For citing our paper, please use:
```
@InProceedings{verbin2020sft,
author = {Verbin, Dor and Zickler, Todd},
title = {Toward a Universal Model for Shape From Texture},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

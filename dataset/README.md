
## Install Dependencies
Conda dependencies can be found in `environment.yaml`
## Synthetic Dataset Generation
To run all the synthetic dataset generation code first:
```
cd dataset
```
To run the image generation code run:
```
python generation.py <dataset_size> <processes> <data_dir>
```
Example:
```
python generation.py 64 16 test
```
Two directories will be created: `line_images` and `line_dict`.`line_dict` is a temporary directory used in `lines.py`. `line_images` will contain all synthetically generated images.

To generate a csv file with img_path and text information run:
```
python lines.py <base_image_dir>
```
Example:
```
python lines.py test
python lines.py
``
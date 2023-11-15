# Readme
## Requirements
Execute the following command to download the dependencies
```
conda install -r requirements.txt
```

## Super Resolution
Download the supaer resolution model from https://huggingface.co/CompVis/ldm-super-resolution-4x-openimages.
Change the path in line28, line 35, line 36 and line 67 in data_super_resolution.py to your local path. 
Execute the following command to run super_resolution
```
python data_super_resolution.py
```

## Color Switching
Execute the file classify.ipynb command to run color switching (change the paths inside as you need)

## Controlled Text-Image Pair Generation
Change the path in line603, line 641 and line 653 in modify.py to your local path. 
Execute the following command to start generation.
```
python modify1/modify.py
```

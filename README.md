# My modifications

To train on yolox `python mmdet_tools/train.py mmdet_configs/xview_yolox/yolox_crop_300_500_cls_60_nopipe.py` 

We have changed the requirments but are yet to update the requirments.txt file. We ran 
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install sahi>=0.9.3
pip install pillow
pip install mmdet==2.21.0
pip install mmcv-full==1.4.3 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
pip install tensorboard>=2.7.0
pip install scipy
pip install setuptools==59.5.0
```
We have also created files to clean and restrict classes to only the trucks and cars,

`python xview/xview_to_coco_cars_trucks.py --train_images_dir="data/xview/train_images" --train_geojson_path="/vol/bitbucket/jrb21/project/xView/data/xView_train.geojson" --output_dir="data/xview/coco" --is_val=True --clean=True`

The last two boolean flags determine whether we are using the validation set or if we are 'cleaining' aka removing negative coordinates which exist in the xview dataset.

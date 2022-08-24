# Before running ths py file execute the following
# module load anaconda3/personal
# module load cuda/11.1.1
# source activate sodb


import os
from PIL import Image
from sahi.model import MmdetDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.utils.cv import visualize_object_predictions
import numpy as np
import pickle
from collections import defaultdict
import datetime


#detection_model = MmdetDetectionModel(
#    model_path= "/rds/general/user/jrb21/home/small-object-detection-benchmark/runs/xview/tood_crop_300_500_cls_cars_trucks_1e-3_new_pipe/latest.pth",
#    config_path= "/rds/general/user/jrb21/home/small-object-detection-benchmark/mmdet_configs/xview_tood/tood_crop_300_500_cls_cars_trucks_1e-3_new_pipe.py",
#    device='cpu' # or 'cpu'
#)


MODEL_PATH = "/rds/general/user/jrb21/home/small-object-detection-benchmark/runs/xview/tood_crop_300_500_cls_cars_trucks_1e-3_new_pipe/latest.pth"
MODEL_CONFIG_PATH = "/rds/general/user/jrb21/home/small-object-detection-benchmark/mmdet_configs/xview_tood/tood_crop_300_500_cls_cars_trucks_1e-3_new_pipe.py"
EVAL_IMAGES_FOLDER_DIR = "WV3/crop_tiff"
EXPORT_VISUAL = True

INFERENCE_SETTING_TO_PARAMS = {
    "XVIEW_SAHI": {
        "no_standard_prediction": True,
        "no_sliced_prediction": False,
        "slice_size": 400,
        "overlap_ratio": 0,
    },
    "XVIEW_SAHI_PO": {
        "no_standard_prediction": True,
        "no_sliced_prediction": False,
        "slice_size": 400,
        "overlap_ratio": 0.20,
    },
    "XVIEW_SAHI_FI": {
        "no_standard_prediction": False,
        "no_sliced_prediction": False,
        "slice_size": 400,
        "overlap_ratio": 0,
    },
    "XVIEW_SAHI_FI_PO": {
        "no_standard_prediction": False,
        "no_sliced_prediction": False,
        "slice_size": 400,
        "overlap_ratio": 0.20,
    },
}

# Of the above 4 options I want to slice and not do full image size inference
INFERENCE_SETTING = "XVIEW_SAHI_PO"
setting_params = INFERENCE_SETTING_TO_PARAMS[INFERENCE_SETTING]

# From the evaluation py file I delete the eval dataset path and change the confidence to 0.3
result = predict(
    model_type="mmdet",
    model_path=MODEL_PATH,
    model_config_path=MODEL_CONFIG_PATH,
    model_confidence_threshold=0.2,
    model_device="cuda:0",
    model_category_mapping=None,
    model_category_remapping=None,
    source=EVAL_IMAGES_FOLDER_DIR,
    no_standard_prediction=setting_params["no_standard_prediction"],
    no_sliced_prediction=setting_params["no_sliced_prediction"],
    image_size=None,
    slice_height=setting_params["slice_size"],
    slice_width=setting_params["slice_size"],
    overlap_height_ratio=setting_params["overlap_ratio"],
    overlap_width_ratio=setting_params["overlap_ratio"],
    postprocess_type="GREEDYNMM",
    postprocess_match_metric="IOS",
    postprocess_match_threshold=0.5,
    postprocess_class_agnostic=True,
    novisual=not EXPORT_VISUAL,
    project="runs/highways",
    name=INFERENCE_SETTING,
    visual_bbox_thickness=1,
    visual_text_size=0.3,
    visual_text_thickness=1,
    visual_export_format="png",
    verbose=0,
    return_dict=True,
    force_postprocess_type=True,
    export_pickle = True,
)

print('success!')

_base_ = ["../yolox/yolox_s_8x8_300e_coco_pipeline.py"]


EXP_NAME = "yolox_crop_300_500_cars_trucks_changed_optimizer_lr_1e-4"
DATA_ROOT = "data/xview/"
BATCH_MULTIPLIER = 2
LR_MULTIPLIER = 1
EVAL_INTERVAL = 3
NUM_CLASSES = 60
DATASET_REPEAT = 50
TAGS = ["yolox", "crop=300_500", "24epochs", f"num_cls={NUM_CLASSES}", f"repeat={DATASET_REPEAT}"]
CLASSES = (
    "Small Car",
    "Bus",
    "Pickup Truck",
    "Utility Truck",
    "Truck",
    "Cargo Truck",
    "Truck w/Box",
    "Truck Tractor",
    "Trailer",
    "Truck w/Flatbed",
    "Truck w/Liquid",
)

NUM_CLASSES = len(CLASSES)

# model settings
model = dict(
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=NUM_CLASSES,
    ),
)

# dataset settings
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomCrop", crop_type="absolute_range", crop_size=(300, 500), allow_negative_crop=True),
    dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]


test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2 * BATCH_MULTIPLIER,
    workers_per_gpu=5,
    train=dict(
        type="RepeatDataset",
        times=DATASET_REPEAT,
        dataset=dict(
            type="CocoDataset",
            classes=CLASSES,
            ann_file=DATA_ROOT + "coco/train_cars_trucks.json",
            img_prefix=DATA_ROOT + "train_images/",
            pipeline=train_pipeline,
        ),
    ),
    val=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "coco/val_cars_trucks.json",
        img_prefix=DATA_ROOT + "train_images/",
        pipeline=test_pipeline,
    ),
    test=dict(
        classes=CLASSES,
        ann_file=DATA_ROOT + "coco/val_cars_trucks.json",
        img_prefix=DATA_ROOT + "train_images/",
        pipeline=test_pipeline,
    ),
)

# optimizer
# default 8 gpu
# /8 for 1 gpu
optimizer = dict(
    type='SGD',
    lr=0.001/8 * BATCH_MULTIPLIER * LR_MULTIPLIER,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)


checkpoint_config = dict(interval=1, max_keep_ckpts=24, save_optimizer=False)
evaluation = dict(interval=EVAL_INTERVAL, metric="bbox", save_best="auto")

# learning policy
# lr_config = dict(policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.1, step=[16, 22])

lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=0,
    min_lr_ratio=0.05)

runner = dict(type="EpochBasedRunner", max_epochs=24)

# logger settings
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook", reset_flag=False),
    ],
)

load_from = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
work_dir = f"runs/xview/{EXP_NAME}/"

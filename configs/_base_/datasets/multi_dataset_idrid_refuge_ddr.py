# dataset settings
dataset_type = 'fundusDataset'

img_scale = (1024, 1024)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='RandomResize', scale=img_scale, ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomRotate',
        prob=1,
        degree=(-15, -11.25, -7.5, -3.75, 0, 3.75, 7.5, 11.25, 15)),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader_1 = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root='../data/IDRiD_vessel',
        data_prefix=dict(
            img_path='train/img', seg_map_path='train/ann'),
        pipeline=train_pipeline))
train_dataloader_2 = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root='../data/FOVCrop-padding/REFUGE-FOVCrop-padding',
        data_prefix=dict(
            img_path='train/images', seg_map_path='train/ann_all'),
        pipeline=train_pipeline))
train_dataloader_3 = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root='../data/FOVCrop-padding/DDR-FOVCrop-padding',
        data_prefix=dict(
            img_path='train/images', seg_map_path='train/ann_all'),
        pipeline=train_pipeline))
train_dataloader = [train_dataloader_1, train_dataloader_2, train_dataloader_3]

val_dataloader_1 = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='../data/FOVCrop-padding/DRIVE-FOVCrop-padding',
        data_prefix=dict(
            img_path='test/images', seg_map_path='test/ann-1st'),
        pipeline=test_pipeline,
        img_suffix='.tif',
        reduce_zero_label=False))
val_dataloader_2 = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='../data/FOVCrop-padding/REFUGE-FOVCrop-padding',
        data_prefix=dict(
            img_path='test/images', seg_map_path='test/ann_all'),
        pipeline=test_pipeline,
        reduce_zero_label=False))
val_dataloader_3 = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='../data/FOVCrop-padding/DDR-FOVCrop-padding',
        data_prefix=dict(
            img_path='test/images', seg_map_path='test/ann_all'),
        pipeline=test_pipeline,
        reduce_zero_label=False))
val_dataloader = [val_dataloader_1, val_dataloader_2, val_dataloader_3]
test_dataloader = val_dataloader

val_evaluator = [dict(type='FundusMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']), dict(type='FundusMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']), dict(type='FundusMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])]
test_evaluator = val_evaluator
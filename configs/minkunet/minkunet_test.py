_base_ = [
    '../_base_/datasets/semantickitti.py', '../_base_/models/minkunet.py',
    '../_base_/schedules/cosine.py', '../_base_/default_runtime.py'
]

model = dict(
    decode_head=dict(
        dropout_ratio=0.,
        loss_decode=dict(type='mmdet.CrossEntropyLoss', avg_non_ignore=True)))
# This schedule is mainly used by models with dynamic voxelization
# optimizer
lr = 2.4e-1  # max learning rate
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='SGD', lr=lr, weight_decay=1.0e-4, momentum=0.9, nesterov=True))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.008, by_epoch=False, begin=0, end=125),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=15,
        by_epoch=True,
        eta_min=0,
        convert_to_iter_based=True)
]
# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=15, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))

randomness = dict(seed=1588147245, deterministic=False, diff_rank_seed=False)
env_cfg = dict(cudnn_benchmark=True)

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti'),
    dict(type='PointSegClassMapping'),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0., 2.0],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

custom_hooks = [dict(type='WorkerInitFnHook')]

train_dataloader = dict(
    num_workers=0,
    sampler=dict(type='DefaultSampler', seed=0, shuffle=True),
    dataset=dict(dataset=dict(pipeline=train_pipeline)))

load_from = 'checkpoints/minkunet_init.pth'

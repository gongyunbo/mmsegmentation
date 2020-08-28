norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type= 'EncoderDecoderDistill',
    teacher_path = 'pspnet_r101-d8_512x1024_40k_cityscapes_20200604_232751-467e7cf4.pth',
    teacher_model=dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))),
    student_model=dict(
    type='EncoderDecoder',
    pretrained='torchvision://resnet18',
    backbone=dict(
        type='ResNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=512,
        in_index=3,
        channels=128,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))),
    distill_loss=[dict(
        type='CriterionPixelWiseLossPPM',
        tau=1.0,
        loss_weight=15.0,
        in_channels = 64,  
        out_channels=256,
    ),dict(
        type='CriterionPixelWiseLossPPM',
        tau=1.0,
        loss_weight=15.0,
        in_channels = 128,  
        out_channels=512,     
    ),dict(
        type='CriterionPixelWiseLossPPM',
        tau=1.0,
        loss_weight=15.0,     
        in_channels = 256,  
        out_channels=1024,
    ),dict(
        type='CriterionPixelWiseLossPPM',
        tau=1.0,
        loss_weight=15.0,     
        in_channels = 512,  
        out_channels=2048,
    ),dict(
        type='CriterionPixelWiseLossLogits',
        tau=1.0,
        loss_weight=15.0,
    )]
    # find_unused_parameters=True
)
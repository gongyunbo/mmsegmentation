_base_ = [
 '../_base_/models/pspnet_r101-distill-r18.py','../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

train_cfg = dict()
test_cfg = dict(mode='whole')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
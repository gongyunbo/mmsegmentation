import torch.nn as nn
import torch.nn.functional as F
import torch

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmcv.runner import  load_checkpoint
from .. import builder
from ..builder import SEGMENTORS_DISTILL
from .base import BaseSegmentor


@SEGMENTORS_DISTILL.register_module()
class EncoderDecoderDistill(BaseSegmentor):
    """Knowledge distillation segmentors.

    It typically consists of teacher_model, student_model.
    """
    def __init__(self,
                 teacher_path,
                 teacher_model,
                 student_model,
                 distill_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,):

        super(EncoderDecoderDistill, self).__init__()
        self.teacher = builder.build_segmentor(teacher_model)
        self.init_weights_teacher(teacher_path)
        self.teacher.eval()
        self.student= builder.build_segmentor(student_model)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.distill_crit_PPM = nn.ModuleList([builder.build_loss(distill_loss[i]) for i in range(len(distill_loss)-1)])
        self.distill_crit_logits = builder.build_loss(distill_loss[-1])


    def init_weights_teacher(self, path=None):
        """Initialize ``teacher_model``"""
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')


    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        with torch.no_grad():
            y_teacher = self.teacher.extract_feat(img)
            seg_logits_teacher = self.teacher.decode_head(y_teacher)
        x_student = self.student.extract_feat(img)
        seg_logits_student = self.student.decode_head(x_student)

        losses = dict()
        #Calculate student  model decode loss
        loss_decode = self.student.decode_head.losses(seg_logits_student,gt_semantic_seg)
        losses.update(loss_decode)
        if self.student.with_auxiliary_head:
            loss_aux = self.student._auxiliary_head_forward_train(
                x_student, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
            
        #Calculate pixel-wise loss of student model and teacher model
        distill_losses_PPM = [self.distill_crit_PPM[i](x_student[i],y_teacher[i]) for i in range(len(x_student))]
        distill_losses_logits = self.distill_crit_logits(seg_logits_student,seg_logits_teacher)
        distill_losses_dict = dict(distill_losses_PPM=distill_losses_PPM,distill_losses_logits=distill_losses_logits)
        losses.update(distill_losses_dict)
        
        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap."""

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.student.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                pad_img = crop_img.new_zeros(
                    (crop_img.size(0), crop_img.size(1), h_crop, w_crop))
                pad_img[:, :, :y2 - y1, :x2 - x1] = crop_img
                pad_seg_logit = self.student.encode_decode(pad_img, img_meta)
                preds[:, :, y1:y2,
                      x1:x2] += pad_seg_logit[:, :, :y2 - y1, :x2 - x1]
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.student.align_corners,
                warning=False)

        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.student.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.student.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        flip_direction = img_meta[0]['flip_direction']
        if flip:
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

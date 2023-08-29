import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from ..util import denormalize_3dlane
import numpy as np


@BBOX_CODERS.register_module()
class LanePseudoCoder(BaseBBoxCoder):

    def __init__(self, denormalize=False):
        self.denormalize = denormalize

    def encode(self):
        pass

    def decode_single(self, cls_scores, lane_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            lane_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """

        cls_scores = cls_scores.sigmoid()
        scores, labels = cls_scores.max(-1)
        if self.denormalize:
            final_lane_preds = denormalize_3dlane(lane_preds, self.pc_range)
        else:
            final_lane_preds = lane_preds

        predictions_dict = {
            'lane3d': final_lane_preds.detach().cpu().numpy(),
            'scores': scores.detach().cpu().numpy(),
            'labels': labels.detach().cpu().numpy()
        }

        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_lanes_preds = preds_dicts['all_lanes_preds'][-1]
        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_lanes_preds[i]))
        return predictions_list

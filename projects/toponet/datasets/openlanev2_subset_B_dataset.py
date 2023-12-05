#---------------------------------------------------------------------------------------#
# Graph-based Topology Reasoning for Driving Scenes (https://arxiv.org/abs/2304.05277)  #
# Source code: https://github.com/OpenDriveLab/TopoNet                                  #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import os
import random
import copy

import numpy as np
import torch
import mmcv
import cv2

from pyquaternion import Quaternion
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset
from openlanev2.evaluation import evaluate as openlanev2_evaluate
from openlanev2.utils import format_metric
from openlanev2.visualization import draw_annotation_pv, assign_attribute, assign_topology

from ..core.lane.util import fix_pts_interpolate
from ..core.visualizer.lane import show_bev_results

from .openlanev2_subset_A_dataset import OpenLaneV2_subset_A_Dataset

@DATASETS.register_module()
class OpenLaneV2_subset_B_Dataset(OpenLaneV2_subset_A_Dataset):
    CAMS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
    MAP_CHANGE_LOGS = [
        'a6daedc3063b421cb3a05019e545f925',
        '02f1e5e2fc544798aad223f5ae5e8440',
        '55638ae3a8b34572aef756ee7fbce0df',
        '20ec831deb0f44e397497198cbe5a97c',
    ]

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information
        """
        info = self.data_infos[index]
        ann_info = info['annotation']

        gt_lanes = [np.array(lane['points'][:, :2], dtype=np.float32) for lane in ann_info['lane_centerline']]
        gt_lane_labels_3d = np.zeros(len(gt_lanes), dtype=np.int64)
        lane_adj = np.array(ann_info['topology_lclc'], dtype=np.float32)

        # only use traffic light attribute
        te_bboxes = np.array([np.array(sign['points'], dtype=np.float32).flatten() for sign in ann_info['traffic_element']])
        te_labels = np.array([sign['attribute'] for sign in ann_info['traffic_element']], dtype=np.int64)
        if len(te_bboxes) == 0:
            te_bboxes = np.zeros((0, 4), dtype=np.float32)
            te_labels = np.zeros((0, ), dtype=np.int64)

        lane_lcte_adj = np.array(ann_info['topology_lcte'], dtype=np.float32)

        assert len(gt_lanes) == lane_adj.shape[0]
        assert len(gt_lanes) == lane_adj.shape[1]
        assert len(gt_lanes) == lane_lcte_adj.shape[0]
        assert len(te_bboxes) == lane_lcte_adj.shape[1]

        annos = dict(
            gt_lanes_3d = gt_lanes,
            gt_lane_labels_3d = gt_lane_labels_3d,
            gt_lane_adj = lane_adj,
            bboxes = te_bboxes,
            labels = te_labels,
            gt_lane_lcte_adj = lane_lcte_adj
        )
        return annos

    def format_results(self, results, jsonfile_prefix=None):
        pred_dict = {}
        pred_dict['method'] = 'TopoNet'
        pred_dict['authors'] = []
        pred_dict['e-mail'] = 'dummy'
        pred_dict['institution / company'] = 'OpenDriveLab'
        pred_dict['country / region'] = 'CN'
        pred_dict['results'] = {}
        for idx, result in enumerate(results):
            info = self.data_infos[idx]
            key = (self.split, info['segment_id'], str(info['timestamp']))

            pred_info = dict(
                lane_centerline=[],
                traffic_element=[],
                topology_lclc=None,
                topology_lcte=None
            )

            if result['lane_results'] is not None:
                lane_results = result['lane_results']
                scores = lane_results[1]
                valid_indices = np.argsort(-scores)
                lanes = lane_results[0][valid_indices]
                lanes = lanes.reshape(-1, lanes.shape[-1] // 2, 2)
                lanes = np.concatenate([lanes, np.zeros_like(lanes[..., 0:1])], axis=-1)

                scores = scores[valid_indices]
                for pred_idx, (lane, score) in enumerate(zip(lanes, scores)):
                    points = fix_pts_interpolate(lane, 11)
                    lc_info = dict(
                        id = 10000 + pred_idx,
                        points = points.astype(np.float32),
                        confidence = score.item()
                    )
                    pred_info['lane_centerline'].append(lc_info)

            if result['bbox_results'] is not None:
                te_results = result['bbox_results']
                scores = te_results[1]
                te_valid_indices = np.argsort(-scores)
                tes = te_results[0][te_valid_indices]
                scores = scores[te_valid_indices]
                class_idxs = te_results[2][te_valid_indices]
                for pred_idx, (te, score, class_idx) in enumerate(zip(tes, scores, class_idxs)):
                    te_info = dict(
                        id = 20000 + pred_idx,
                        category = 1 if class_idx < 4 else 2,
                        attribute = class_idx,
                        points = te.reshape(2, 2).astype(np.float32),
                        confidence = score
                    )
                    pred_info['traffic_element'].append(te_info)

            if result['lclc_results'] is not None:
                pred_info['topology_lclc'] = result['lclc_results'].astype(np.float32)[valid_indices][:, valid_indices]
            else:
                pred_info['topology_lclc'] = np.zeros((len(pred_info['lane_centerline']), len(pred_info['lane_centerline'])), dtype=np.float32)

            if result['lcte_results'] is not None:
                pred_info['topology_lcte'] = result['lcte_results'].astype(np.float32)[valid_indices][:, te_valid_indices]
            else:
                pred_info['topology_lcte'] = np.zeros((len(pred_info['lane_centerline']), len(pred_info['traffic_element'])), dtype=np.float32)

            pred_dict['results'][key] = dict(predictions=pred_info)

        return pred_dict

    @staticmethod
    def _render_surround_img(images):
        all_image = []
        img_height = images[1].shape[0]

        for idx in [2, 0, 1, 5, 3, 4]:
            if idx == 4 or idx == 1:
                all_image.append(images[idx])
            else:
                all_image.append(images[idx])
                all_image.append(np.full((img_height, 10, 3), (255, 255, 255), dtype=np.uint8))

        surround_img_upper = None
        surround_img_upper = np.concatenate(all_image[:5], 1)

        surround_img_down = None
        surround_img_down = np.concatenate(all_image[5:], 1)

        surround_img = np.concatenate((surround_img_upper, np.full((10, surround_img_down.shape[1], 3), (255, 255, 255), dtype=np.uint8), surround_img_down), 0)
        surround_img = cv2.resize(surround_img, None, fx=0.5, fy=0.5)

        return surround_img

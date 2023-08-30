#---------------------------------------------------------------------------------------#
# Graph-based Topology Reasoning for Driving Scenes (https://arxiv.org/abs/2304.05277)  #
# Source code: https://github.com/OpenDriveLab/TopoNet                                  #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
from mmdet.datasets.builder import PIPELINES
from shapely.geometry import LineString
from ...core.lane.util import fix_pts_interpolate


@PIPELINES.register_module()
class LaneParameterize3D(object):

    def __init__(self, method, method_para):
        method_list = ['fix_pts_interp']
        self.method = method
        if not self.method in method_list:
            raise Exception("Not implemented!")
        self.method_para = method_para

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        lanes = results['gt_lanes_3d']
        para_lanes = getattr(self, self.method)(lanes, **self.method_para)
        results['gt_lanes_3d'] = para_lanes

        return results

    def fix_pts_interp(self, input_data, n_points=11):
        '''Interpolate the 3D lanes to fix points. The input size is (n_pts, 3).
        '''
        lane_list = []
        for lane in input_data:
            if n_points == 11 and lane.shape[0] == 201:
                lane_list.append(lane[::20].flatten())
            else:
                lane = fix_pts_interpolate(lane, n_points).flatten()
                lane_list.append(lane)
        return np.array(lane_list, dtype=np.float32)


@PIPELINES.register_module()
class LaneLengthFilter(object):
    """Filter the 3D lanes by lane length (meters).
    """

    def __init__(self, min_length):
        self.min_length = min_length

    def __call__(self, results):

        if self.min_length <= 0:
            return results

        length_list = np.array(list(map(lambda x:LineString(x).length, results['gt_lanes_3d'])))
        masks = length_list > self.min_length
        results['gt_lanes_3d'] = [lane for idx, lane in enumerate(results['gt_lanes_3d']) if masks[idx]]
        results['gt_lane_labels_3d'] = results['gt_lane_labels_3d'][masks]

        if 'gt_lane_adj' in results.keys():
            results['gt_lane_adj'] = results['gt_lane_adj'][masks][:, masks]
        if 'gt_lane_lcte_adj' in results.keys():
            results['gt_lane_lcte_adj'] = results['gt_lane_lcte_adj'][masks]

        return results

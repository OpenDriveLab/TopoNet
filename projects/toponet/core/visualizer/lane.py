import numpy as np
import mmcv
import cv2
from openlanev2.visualization.utils import COLOR_DICT

GT_COLOR = (0, 255, 0)
PRED_COLOR = (44, 63, 255)


def show_results(image_list, lidar2imgs, gt_lane, pred_lane, gt_te=None, pred_te=None):
    res_image_list = []
    for idx, (raw_img, lidar2img) in enumerate(zip(image_list, lidar2imgs)):
        image = raw_img.copy()
        for lane in gt_lane:
            xyz1 = np.concatenate([lane, np.ones((lane.shape[0], 1))], axis=1)
            xyz1 = xyz1 @ lidar2img.T
            xyz1 = xyz1[xyz1[:, 2] > 1e-5]
            if xyz1.shape[0] == 0:
                continue
            points_2d = xyz1[:, :2] / xyz1[:, 2:3]
            points_2d = points_2d.astype(int)
            image = cv2.polylines(image, points_2d[None], False, GT_COLOR, 2)

        for lane in pred_lane:
            xyz1 = np.concatenate([lane, np.ones((lane.shape[0], 1))], axis=1)
            xyz1 = xyz1 @ lidar2img.T
            xyz1 = xyz1[xyz1[:, 2] > 1e-5]
            if xyz1.shape[0] == 0:
                continue
            points_2d = xyz1[:, :2] / xyz1[:, 2:3]
            points_2d = points_2d.astype(int)
            image = cv2.polylines(image, points_2d[None], False, PRED_COLOR, 2)

        if idx == 0:
            if gt_te is not None:
                for bbox, attr in gt_te:
                    b = bbox.astype(int)
                    color = COLOR_DICT[attr]
                    image = draw_corner_rectangle(image, (b[0], b[1]), (b[2], b[3]), color, 3, 1)
            if pred_te is not None:
                for bbox, attr in pred_te:
                    b = bbox.astype(int)
                    color = COLOR_DICT[attr]
                    image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, 3)

        res_image_list.append(image)

    return res_image_list

def show_bev_results(gt_lane, pred_lane, gt_lclc=None, pred_lclc=None, only=None, map_size=[-55, 55, -30, 30], scale=10):
    image = np.zeros((int(scale*(map_size[1]-map_size[0])), int(scale*(map_size[3] - map_size[2])), 3), dtype=np.uint8)
    if only is None or only == 'gt':
        for lane in gt_lane:
            draw_coor = (scale * (-lane[:, :2] + np.array([map_size[1], map_size[3]]))).astype(int)
            image = cv2.polylines(image, [draw_coor[:, [1,0]]], False, GT_COLOR, max(round(scale * 0.2), 1))
            image = cv2.circle(image, (draw_coor[0, 1], draw_coor[0, 0]), max(2, round(scale * 0.5)), GT_COLOR, -1)
            image = cv2.circle(image, (draw_coor[-1, 1], draw_coor[-1, 0]), max(2, round(scale * 0.5)) , GT_COLOR, -1)
        
        if gt_lclc is not None:
            for l1_idx, lclc in enumerate(gt_lclc):
                for l2_idx, connected in enumerate(lclc):
                    if connected:
                        l1 = gt_lane[l1_idx]
                        l2 = gt_lane[l2_idx]
                        l1_mid = len(l1) // 2
                        l2_mid = len(l2) // 2
                        p1 = (scale * (-l1[l1_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(int)
                        p2 = (scale * (-l2[l2_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(int)
                        image = cv2.arrowedLine(image, (p1[1], p1[0]), (p2[1], p2[0]), GT_COLOR, max(round(scale * 0.1), 1), tipLength=0.1)
    
    if only is None or only == 'pred':
        for lane in pred_lane:
            draw_coor = (scale * (-lane[:, :2] + np.array([map_size[1], map_size[3]]))).astype(int)
            image = cv2.polylines(image, [draw_coor[:, [1,0]]], False, PRED_COLOR, max(round(scale * 0.2), 1))
            image = cv2.circle(image, (draw_coor[0, 1], draw_coor[0, 0]), max(2, round(scale * 0.5)), PRED_COLOR, -1)
            image = cv2.circle(image, (draw_coor[-1, 1], draw_coor[-1, 0]), max(2, round(scale * 0.5)) , PRED_COLOR, -1)

        if pred_lclc is not None:
            for l1_idx, lclc in enumerate(pred_lclc):
                for l2_idx, connected in enumerate(lclc):
                    if connected:
                        l1 = pred_lane[l1_idx]
                        l2 = pred_lane[l2_idx]
                        l1_mid = len(l1) // 2
                        l2_mid = len(l2) // 2
                        p1 = (scale * (-l1[l1_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(int)
                        p2 = (scale * (-l2[l2_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(int)
                        image = cv2.arrowedLine(image, (p1[1], p1[0]), (p2[1], p2[0]), PRED_COLOR, max(round(scale * 0.1), 1), tipLength=0.1)

    return image

def draw_corner_rectangle(img: np.ndarray, pt1: tuple, pt2: tuple, color: tuple,
        corner_thickness: int = 3, edge_thickness: int = 2,
        centre_cross: bool = False, lineType: int = cv2.LINE_8):

    corner_length = min(abs(pt1[0] - pt2[0]), abs(pt1[1] - pt2[1])) // 4
    e_args = [color, edge_thickness, lineType]
    c_args = [color, corner_thickness, lineType]

    # edges
    img = cv2.line(img, (pt1[0] + corner_length, pt1[1]), (pt2[0] - corner_length, pt1[1]), *e_args)
    img = cv2.line(img, (pt2[0], pt1[1] + corner_length), (pt2[0], pt2[1] - corner_length), *e_args)
    img = cv2.line(img, (pt1[0], pt1[1] + corner_length), (pt1[0], pt2[1] - corner_length), *e_args)
    img = cv2.line(img, (pt1[0] + corner_length, pt2[1]), (pt2[0] - corner_length, pt2[1]), *e_args)
    # corners
    img = cv2.line(img, pt1, (pt1[0] + corner_length, pt1[1]), *c_args)
    img = cv2.line(img, pt1, (pt1[0], pt1[1] + corner_length), *c_args)
    img = cv2.line(img, (pt2[0], pt1[1]), (pt2[0] - corner_length, pt1[1]), *c_args)
    img = cv2.line(img, (pt2[0], pt1[1]), (pt2[0], pt1[1] + corner_length), *c_args)
    img = cv2.line(img, (pt1[0], pt2[1]), (pt1[0] + corner_length, pt2[1]), *c_args)
    img = cv2.line(img, (pt1[0], pt2[1]), (pt1[0], pt2[1] - corner_length), *c_args)
    img = cv2.line(img, pt2, (pt2[0] - corner_length, pt2[1]), *c_args)
    img = cv2.line(img, pt2, (pt2[0], pt2[1] - corner_length), *c_args)

    if centre_cross:
        cx, cy = int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2)
        img = cv2.line(img, (cx - corner_length, cy), (cx + corner_length, cy), *e_args)
        img = cv2.line(img, (cx, cy - corner_length), (cx, cy + corner_length), *e_args)
    
    return img

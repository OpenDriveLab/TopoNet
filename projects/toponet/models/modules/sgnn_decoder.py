#---------------------------------------------------------------------------------------#
# Graph-based Topology Reasoning for Driving Scenes (https://arxiv.org/abs/2304.05277)  #
# Source code: https://github.com/OpenDriveLab/TopoNet                                  #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import copy
import warnings
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.cnn import Linear, build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout 
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER, FEEDFORWARD_NETWORK,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TransformerLayerSequence
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TopoNetSGNNDecoder(TransformerLayerSequence):

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(TopoNetSGNNDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                *args,
                reference_points=None,
                lclc_branches=None,
                lcte_branches=None,
                key_padding_mask=None,
                te_feats=None,
                te_cls_scores=None,
                **kwargs):

        output = query
        intermediate = []
        intermediate_reference_points = []
        intermediate_lclc_rel = []
        intermediate_lcte_rel = []
        num_query = query.size(0)
        num_te_query = te_feats.size(2)

        prev_lclc_adj = torch.zeros((query.size(1), num_query, num_query),
                                  dtype=query.dtype, device=query.device)
        prev_lcte_adj = torch.zeros((query.size(1), num_query, num_te_query),
                                  dtype=query.dtype, device=query.device)
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                te_query=te_feats[lid],
                te_cls_scores=te_cls_scores[lid],
                lclc_adj=prev_lclc_adj,
                lcte_adj=prev_lcte_adj,
                **kwargs)
            output = output.permute(1, 0, 2)

            lclc_rel_out = lclc_branches[lid](output, output)
            lclc_rel_adj = lclc_rel_out.squeeze(-1).sigmoid()
            prev_lclc_adj = lclc_rel_adj.detach()

            lcte_rel_out = lcte_branches[lid](output, te_feats[lid])
            lcte_rel_adj = lcte_rel_out.squeeze(-1).sigmoid()
            prev_lcte_adj = lcte_rel_adj.detach()

            output = output.permute(1, 0, 2)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_lclc_rel.append(lclc_rel_out)
                intermediate_lcte_rel.append(lcte_rel_out)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points), torch.stack(
                intermediate_lclc_rel), torch.stack(
                intermediate_lcte_rel)

        return output, reference_points, lclc_rel_out, lcte_rel_out


@TRANSFORMER_LAYER.register_module()
class SGNNDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 ffn_cfgs,
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 **kwargs):
        super(SGNNDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
    
    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                te_query=None,
                te_cls_scores=None,
                lclc_adj=None,
                lcte_adj=None,
                **kwargs):

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, te_query, lclc_adj, lcte_adj, te_cls_scores, identity=identity if self.pre_norm else None)
                ffn_index += 1

        return query


@FEEDFORWARD_NETWORK.register_module()
class FFN_SGNN(BaseModule):

    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=512,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.1,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 edge_weight=0.5, 
                 num_te_classes=13,
                 **kwargs):
        super(FFN_SGNN, self).__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(
            Sequential(
                Linear(feedforward_channels, embed_dims), self.activate,
                nn.Dropout(ffn_drop)))
        self.layers = Sequential(*layers)
        self.edge_weight = edge_weight

        self.lclc_gnn_layer = LclcSkgGCNLayer(embed_dims, embed_dims, edge_weight=edge_weight)
        self.lcte_gnn_layer = LcteSkgGCNLayer(embed_dims, embed_dims, num_te_classes=num_te_classes, edge_weight=edge_weight)

        self.downsample = nn.Linear(embed_dims * 2, embed_dims)

        self.gnn_dropout1 = nn.Dropout(ffn_drop)
        self.gnn_dropout2 = nn.Dropout(ffn_drop)

        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, lc_query, te_query, lclc_adj, lcte_adj, te_cls_scores, identity=None):

        out = self.layers(lc_query)
        out = out.permute(1, 0, 2)

        lclc_features = self.lclc_gnn_layer(out, lclc_adj)
        lcte_features = self.lcte_gnn_layer(te_query, lcte_adj, te_cls_scores)

        out = torch.cat([lclc_features, lcte_features], dim=-1)

        out = self.activate(out)
        out = self.gnn_dropout1(out)
        out = self.downsample(out)
        out = self.gnn_dropout2(out)

        out = out.permute(1, 0, 2)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = lc_query
        return identity + self.dropout_layer(out)


class LclcSkgGCNLayer(nn.Module):

    def __init__(self, in_features, out_features, edge_weight=0.5):
        super(LclcSkgGCNLayer, self).__init__()
        self.edge_weight = edge_weight

        if self.edge_weight != 0:
            self.weight_forward = torch.Tensor(in_features, out_features)
            self.weight_forward = nn.Parameter(nn.init.xavier_uniform_(self.weight_forward))
            self.weight_backward = torch.Tensor(in_features, out_features)
            self.weight_backward = nn.Parameter(nn.init.xavier_uniform_(self.weight_backward))

        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        self.edge_weight = edge_weight

    def forward(self, input, adj):

        support_loop = torch.matmul(input, self.weight)
        output = support_loop

        if self.edge_weight != 0:
            support_forward = torch.matmul(input, self.weight_forward)
            output_forward = torch.matmul(adj, support_forward)
            output += self.edge_weight * output_forward

            support_backward = torch.matmul(input, self.weight_backward)
            output_backward = torch.matmul(adj.permute(0, 2, 1), support_backward)
            output += self.edge_weight * output_backward

        return output


class LcteSkgGCNLayer(nn.Module):

    def __init__(self, in_features, out_features, num_te_classes=13, edge_weight=0.5):
        super(LcteSkgGCNLayer, self).__init__()
        self.weight = torch.Tensor(num_te_classes, in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        self.edge_weight = edge_weight

    def forward(self, te_query, lcte_adj, te_cls_scores):
        # te_cls_scores: (bs, num_te_query, num_te_classes)
        cls_scores = te_cls_scores.detach().sigmoid().unsqueeze(3)
        # te_query: (bs, num_te_query, embed_dims)
        # (bs, num_te_query, 1, embed_dims) * (bs, num_te_query, num_te_classes, 1)
        te_feats = te_query.unsqueeze(2) * cls_scores
        # (bs, num_te_classes, num_te_query, embed_dims)
        te_feats = te_feats.permute(0, 2, 1, 3)

        support = torch.matmul(te_feats, self.weight).sum(1)
        adj = lcte_adj * self.edge_weight
        output = torch.matmul(adj, support)
        return output

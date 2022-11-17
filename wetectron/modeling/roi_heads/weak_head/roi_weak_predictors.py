# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F

from wetectron.modeling import registry

import math
###
from wetectron.structures.bounding_box import BoxList, BatchBoxList
from wetectron.structures.boxlist_ops import boxlist_iou, boxlist_iou_async
from wetectron.utils.utils import to_boxlist, cal_iou, cal_adj, normalize_graph
from wetectron.modeling.roi_heads.sim_head.sim_head import Sim_Net
###
from torch_geometric.nn import GCNConv, BatchNorm, GATConv, global_add_pool, global_mean_pool, global_max_pool

@registry.ROI_WEAK_PREDICTOR.register("GCNPredictor")
class GCNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(GCNPredictor, self).__init__()
        assert in_channels is not None
        num_inputs = in_channels
        gcn_num_inputs = 512
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes


        ###
        hidden_layer = 42
        #self.gcn_cls1 = nn.Linear(gcn_num_inputs, hidden_layer)
        #self.gcn_cls2 = nn.Linear(hidden_layer, num_classes)
        #self.gcn_det1 = nn.Linear(gcn_num_inputs, hidden_layer)
        #self.gcn_det2 = nn.Linear(hidden_layer, num_classes)

        '''self.ref1 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred1 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        self.ref2 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred2 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        self.ref3 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred3 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)
        '''
        self.gcn_cls1 = GCNConv(
            in_channels=num_inputs, out_channels=hidden_layer, add_self_loops=False, normalize=False)
        self.gcn_cls2 = GCNConv(
            in_channels=hidden_layer, out_channels=num_classes, add_self_loops=False, normalize=False)

        self.gcn_det1 = GCNConv(
            in_channels=num_inputs, out_channels=hidden_layer, add_self_loops=False, normalize=False)
        self.gcn_det2 = GCNConv(
            in_channels=hidden_layer, out_channels=num_classes, add_self_loops=False, normalize=False)

        #self.linear_img = nn.Linear(hidden_layer2, num_classes)
        #self.linear_2 = nn.Linear(hidden_layer2, num_classes)
        self.bn = BatchNorm(hidden_layer)
        #self._initialize_weights()

        #self.model_sim = Sim_Net(config, in_channels)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #std = 1. / math.sqrt(m.weight.size(1))
                #nn.init.uniform_(m.weight, -std, std)
                #nn.init.uniform_(m.bias, -std, std)
                #nn.init.normal_(m.weight, mean=0, std=0.001)
                #nn.init.constant_(m.bias, 0)
                #nn.init.xavier_uniform_(m.weight)
                #nn.init.constant_(m.bias, 0)
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
                #import IPython; IPython.embed()
                #if m.in_features == 512 or m.in_features == 42:
                #    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                #    nn.init.constant_(m.bias, 0)
                #else:
                #    nn.init.normal_(m.weight, mean=0, std=0.001)
                #    nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, proposals):
        if x2.dim() == 4:
            x2 = self.avgpool(x2)
            x2 = x2.view(x2.size(0), -1)
        assert x2.dim() == 2
        #cls = self.cls_score(x)
        #det = self.det_score(x)

        l_proposal = [len(proposal) for proposal in proposals]
        x_list = x1.split([len(p) for p in proposals])
        cls_logit = torch.zeros((0, self.num_classes), dtype=torch.float, device=x1.device)
        det_logit = torch.zeros((0, self.num_classes), dtype=torch.float, device=x1.device)
        img_logit = []


        for (proposal, x_per_img) in zip(proposals, x_list):
            adj_m = cal_adj(proposal)
            adj_m[adj_m < 0.5] = 0
            #adj_m = adj_m - torch.eye(len(proposal)).to(x1.device)
            #adj_degree_m = torch.pow(torch.cholesky_inverse(torch.diag(torch.sum(adj_m, 1))), 1/2)

            #sim_feature = self.model_sim(x_per_img).detach() ### adjacency matrix should not be learnable
            sim_feature = F.normalize(x_per_img,dim=1)
            sim_m = torch.mm(sim_feature, sim_feature.T)
            sim_m[torch.isnan(sim_m)] = 0
            sim_m[sim_m < 0.5] = 0
            #sim_m = sim_m - torch.eye(len(proposal)).to(x1.device)
            #sim_degree_m = torch.pow(torch.cholesky_inverse(torch.diag(torch.sum(sim_m, 1))), 1/2)
            #sim_degree_m[torch.isnan(sim_degree_m)] = 0

            #norm_x = x_per_img
            x = torch.Tensor(normalize_graph(x_per_img.cpu().detach())).to(x1.device)

            #norm_adj_m = torch.mm(torch.mm(adj_degree_m, adj_m), adj_degree_m)
            norm_adj_m = torch.Tensor(normalize_graph(adj_m.cpu().detach())).to(x1.device)

            #norm_sim_m = torch.mm(torch.mm(sim_degree_m, sim_m), sim_degree_m)
            norm_sim_m = torch.Tensor(normalize_graph(sim_m.cpu().detach())).to(x1.device)

            edge_index  = adj_m.nonzero(as_tuple=False).T
            #edge_weight = norm_adj_m[adj_m.nonzero(as_tuple=True)].unsqueeze(1)
            #edge_weight.requires_grad = True

            cls_out = self.gcn_cls1(x, edge_index)#, edge_weight)
            cls_out = F.dropout(F.relu(cls_out), training=self.training)
            cls_out = self.gcn_cls2(cls_out, edge_index)#, edge_weight)

            #cls_out = torch.spmm(norm_sim_m, self.gcn_cls1(norm_x))
            #cls_out = F.dropout(F.relu(cls_out), training=self.training)
            #cls_out = torch.spmm(norm_sim_m, self.gcn_cls2(cls_out))

            edge_index  = sim_m.nonzero(as_tuple=False).T
            #edge_weight = norm_sim_m[sim_m.nonzero(as_tuple=True)].unsqueeze(1)
            #edge_weight.requires_grad = True

            det_out = self.gcn_det1(x, edge_index)#, edge_weight)
            det_out = F.dropout(F.relu(det_out), training=self.training)
            det_out = self.gcn_det2(det_out, edge_index)#, edge_weight)

            #det_out = torch.spmm(norm_adj_m, self.gcn_det1(norm_x))
            #det_out = F.dropout(F.relu(det_out), training=self.training)
            #det_out = torch.spmm(norm_adj_m, self.gcn_det2(det_out))

            #import IPython; IPython.embed()

            cls_logit = torch.cat((cls_logit, cls_out))
            det_logit = torch.cat((det_logit, det_out))

            #img_out = F.softmax((cls_out * det_out).mean(0))
            #img_logit.append(img_out)
        #cls_logit_list = cls_logit.split([len(p) for p in proposals])
        #det_logit_list = det_logit.split([len(p) for p in proposals])
        #import IPython; IPython.embed()
        #for cls_per_img, det_per_img in zip(cls_logit_list, det_logit_list):
        #    img_logit.append(F.softmax((cls_per_img + det_per_img).sum(0)))

        #img_logit = F.softmax((cls_logit + det_logit).sum(0))
        #import IPython; IPython.embed()
        '''ref1_logit = self.ref1(x1)
        bbox_pred1 = self.bbox_pred1(x1)
        ref2_logit = self.ref2(x1)
        bbox_pred2 = self.bbox_pred2(x1)
        ref3_logit = self.ref3(x1)
        bbox_pred3 = self.bbox_pred3(x1)
        '''
        if not self.training:
            #cls_logit = self.gcn_cls(x)
            #det_logit = self.gcn_det(x)
            cls_logit = F.softmax(cls_logit, dim=1)
            # do softmax along ROI for different imgs
            det_logit_list = det_logit.split([len(p) for p in proposals])
            final_det_logit = []
            for det_logit_per_image in det_logit_list:
                det_logit_per_image = F.softmax(det_logit_per_image, dim=0)
                final_det_logit.append(det_logit_per_image)
            final_det_logit = torch.cat(final_det_logit, dim=0)
        else:
            final_det_logit = det_logit

        #ref_logits = [ref1_logit, ref2_logit, ref3_logit]
        #bbox_preds = [bbox_pred1, bbox_pred2, bbox_pred3]

        return cls_logit, final_det_logit, img_logit #ref_logits, bbox_preds, img_logit

@registry.ROI_WEAK_PREDICTOR.register("WSDDNPredictor")
class WSDDNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(WSDDNPredictor, self).__init__()
        assert in_channels is not None
        num_inputs = in_channels
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.det_score = nn.Linear(num_inputs, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, proposals):
        if x.dim() == 4:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        assert x.dim() == 2

        cls_logit = self.cls_score(x)
        det_logit = self.det_score(x)

        if not self.training:
            cls_logit = F.softmax(cls_logit, dim=1)
            # do softmax along ROI for different imgs
            det_logit_list = det_logit.split([len(p) for p in proposals])
            final_det_logit = []
            for det_logit_per_image in det_logit_list:
                det_logit_per_image = F.softmax(det_logit_per_image, dim=0)
                final_det_logit.append(det_logit_per_image)
            final_det_logit = torch.cat(final_det_logit, dim=0)
        else:
            final_det_logit = det_logit

        return cls_logit, final_det_logit, None


@registry.ROI_WEAK_PREDICTOR.register("OICRPredictor")
class OICRPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(OICRPredictor, self).__init__()
        assert in_channels is not None
        num_inputs = in_channels
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.det_score = nn.Linear(num_inputs, num_classes)

        self.ref1 = nn.Linear(num_inputs, num_classes)
        self.ref2 = nn.Linear(num_inputs, num_classes)
        self.ref3 = nn.Linear(num_inputs, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, proposals):
        if x.dim() == 4:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        assert x.dim() == 2

        cls_logit = self.cls_score(x)
        det_logit = self.det_score(x)
        ref1_logit = self.ref1(x)
        ref2_logit = self.ref2(x)
        ref3_logit = self.ref3(x)

        if not self.training:
            cls_logit = F.softmax(cls_logit, dim=1)
            # do softmax along ROI for different imgs
            det_logit_list = det_logit.split([len(p) for p in proposals])
            final_det_logit = []
            for det_logit_per_image in det_logit_list:
                det_logit_per_image = F.softmax(det_logit_per_image, dim=0)
                final_det_logit.append(det_logit_per_image)
            final_det_logit = torch.cat(final_det_logit, dim=0)
            #
            ref1_logit = F.softmax(ref1_logit, dim=1)
            ref2_logit = F.softmax(ref2_logit, dim=1)
            ref3_logit = F.softmax(ref3_logit, dim=1)
        else:
            final_det_logit = det_logit

        ref_logits = [ref1_logit, ref2_logit, ref3_logit]
        return cls_logit, final_det_logit, ref_logits


@registry.ROI_WEAK_PREDICTOR.register("MISTPredictor")
class MISTPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MISTPredictor, self).__init__()
        assert in_channels is not None
        num_inputs = in_channels
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.det_score = nn.Linear(num_inputs, num_classes)

        self.ref1 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred1 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        self.ref2 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred2 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        self.ref3 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred3 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, proposals):
        if x.dim() == 4:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        assert x.dim() == 2

        cls_logit = self.cls_score(x)
        det_logit = self.det_score(x)
        ref1_logit = self.ref1(x)
        bbox_pred1 = self.bbox_pred1(x)
        ref2_logit = self.ref2(x)
        bbox_pred2 = self.bbox_pred2(x)
        ref3_logit = self.ref3(x)
        bbox_pred3 = self.bbox_pred3(x)

        if not self.training:
            cls_logit = F.softmax(cls_logit, dim=1)
            # do softmax along ROI for different imgs
            det_logit_list = det_logit.split([len(p) for p in proposals])
            final_det_logit = []
            for det_logit_per_image in det_logit_list:
                det_logit_per_image = F.softmax(det_logit_per_image, dim=0)
                final_det_logit.append(det_logit_per_image)
            final_det_logit = torch.cat(final_det_logit, dim=0)
            ref1_logit = F.softmax(ref1_logit, dim=1)
            ref2_logit = F.softmax(ref2_logit, dim=1)
            ref3_logit = F.softmax(ref3_logit, dim=1)
        else:
            final_det_logit = det_logit

        ref_logits = [ref1_logit, ref2_logit, ref3_logit]
        bbox_preds = [bbox_pred1, bbox_pred2, bbox_pred3]
        return cls_logit, final_det_logit, ref_logits, bbox_preds


def make_roi_weak_predictor(cfg, in_channels):
    func = registry.ROI_WEAK_PREDICTOR[cfg.MODEL.ROI_WEAK_HEAD.PREDICTOR]
    return func(cfg, in_channels)

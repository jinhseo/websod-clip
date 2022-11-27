# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
import torch
import torch.nn.functional as F

from wetectron.modeling import registry
from torch_geometric.nn import aggr
from torch import nn

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
        num_inputs = 2048
        gcn_num_inputs = 512
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes


        ###
        #hidden_layer = 512
        self.ref1 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred1 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        self.ref2 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred2 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        self.ref3 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred3 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        '''self.gcn_cls1 = GCNConv(
            in_channels=gcn_num_inputs, out_channels=hidden_layer, add_self_loops=False, normalize=False)
        self.gcn_cls2 = GCNConv(
            in_channels=hidden_layer, out_channels=num_classes, add_self_loops=False, normalize=False)

        self.gcn_det1 = GCNConv(
            in_channels=gcn_num_inputs, out_channels=hidden_layer, add_self_loops=False, normalize=False)
        self.gcn_det2 = GCNConv(
            in_channels=hidden_layer, out_channels=num_classes, add_self_loops=False, normalize=False)
        '''
        #hidden_layer = 512
        #num_heads = 1
        #self.gat_1 = GATConv(num_inputs, hidden_layer, num_heads, add_self_loops=True, concat=False,  dropout=0.0) ### pseudo labeling during training
        #self.gat_2 = GATConv(hidden_layer, 42, heads=1, add_self_loops=True, concat=False, dropout=0.0)

        hidden_layer = 8
        num_heads = 8
        num_heads2 = 1
        self.gat_1 = GATConv(num_inputs, hidden_layer, num_heads, add_self_loops=True, concat=True,  dropout=0.0)
        #self.gat_2 = GATConv(hidden_layer*num_heads, 4096, heads=1, add_self_loops=False, concat=False, dropout=0.0)
        self.gat_2 = GATConv(hidden_layer * num_heads, num_classes, num_heads2, add_self_loops=True, concat=True,  dropout=0.0)

        #self.node_linear = nn.Linear(hidden_layer * num_heads2, num_classes)
        #self.graph_linear = nn.Linear(42, num_classes)
        #self.gat_det1 = GATConv(num_inputs, hidden_layer, num_heads, add_self_loops=True,concat=False, dropout=0.0)
        #self.gat_det2 = GATConv(hidden_layer, num_classes, heads=1, add_self_loops=True,concat=False, dropout=0.0)
        self.mean_aggr = aggr.MeanAggregation()
        self.soft_aggr = aggr.SoftmaxAggregation(learn=True)
        #self.bn = BatchNorm(hidden_layer)
        self._initialize_weights()

        self.model_sim = Sim_Net(num_inputs)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, proposals, targets):
        if x2.dim() == 4:
            x2 = self.avgpool(x2)
            x2 = x2.view(x2.size(0), -1)
        assert x2.dim() == 2

        device = x1.device
        l_proposal = [len(proposal) for proposal in proposals]
        x_list = x1.split([len(p) for p in proposals])
        #cls_logit = torch.zeros((0, self.num_classes), dtype=torch.float, device=device)
        #det_logit = torch.zeros((0, self.num_classes), dtype=torch.float, device=device)
        #graph_score_per_node = torch.zeros((self.num_classes), dtype=torch.float, device=device)

        graph_score_list = []
        node_score_list  = []

        for (proposal, target, x_per_img) in zip(proposals, targets, x_list):
            x = x_per_img
            target_label = int(target.get_field('labels').item())
            #x = x.view(x.shape[0], -1)

            adj_m = cal_adj(proposal)
            adj_m[adj_m < 0.4] = 0
            #adj_m = adj_m - torch.eye(len(proposal)).to(x1.device)
            #adj_degree_m = torch.pow(torch.cholesky_inverse(torch.diag(torch.sum(adj_m, 1))), 1/2)
            import IPython; IPython.embed()
            sim_feature = self.model_sim(x)
            #sim_feature = F.normalize(x, dim=1)
            sim_m = torch.mm(sim_feature, sim_feature.T)
            sim_m[sim_m < 0.4] = 0
            #sim_m = sim_m - torch.eye(len(proposal)).to(x1.device)
            #sim_degree_m = torch.pow(torch.cholesky_inverse(torch.diag(torch.sum(sim_m, 1))), 1/2)
            #sim_degree_m[torch.isnan(sim_degree_m)] = 0

            #x = x_per_img
            #x = torch.Tensor(normalize_graph(x_per_img.cpu().detach())).to(device)

            #norm_adj_m = torch.mm(torch.mm(adj_degree_m, adj_m), adj_degree_m)
            #norm_adj_m = torch.Tensor(normalize_graph(adj_m.cpu().detach())).to(device)
            #norm_sim_m = torch.mm(torch.mm(sim_degree_m, sim_m), sim_degree_m)
            #norm_sim_m = torch.Tensor(normalize_graph(sim_m.cpu().detach())).to(device)

            '''edge_index = adj_m.nonzero(as_tuple=False).T
            #edge_weight = norm_adj_m[adj_m.nonzero(as_tuple=True)].unsqueeze(1)

            #cls_out = self.gcn_cls1(x, edge_index, edge_weight)
            #cls_out = F.dropout(F.relu(cls_out), training=self.training)
            #cls_out = self.gcn_cls2(cls_out, edge_index, edge_weight)
            '''

            iou_edge = adj_m.nonzero(as_tuple=False).T
            sim_edge = sim_m.nonzero(as_tuple=False).T
            #x = F.dropout(x, p=0.6, training=self.training)
            node_out, (a_ind, a_weight) = self.gat_1(x, iou_edge, return_attention_weights=True)
            node_out = F.dropout(F.elu(node_out), p=0.6, training=self.training)
            node_out, (a_ind2, a_weight2) = self.gat_2(node_out, sim_edge, return_attention_weights=True)
            #node_out = F.dropout(F.elu(node_out), p=0.6, training=self.training)
            '''#edge_index  = sim_m.nonzero(as_tuple=False).T
            #edge_weight = norm_sim_m[sim_m.nonzero(as_tuple=True)].unsqueeze(1)
            #det_out = self.gcn_det1(x, edge_index, edge_weight)
            #det_out = F.dropout(F.relu(det_out), training=self.training)
            #det_out = self.gcn_det2(det_out, edge_index, edge_weight)
            '''
            a_matrix = torch.zeros_like(adj_m)
            if a_weight.dim() != 1:
                a_weight = a_weight.mean(1)
            if a_weight2.dim() != 1 :
                a_weight2 = a_weight2.mean(1)
            #attention_weight = (a_weight + a_weight2)/2
            #a_matrix[a_ind[0,:], a_ind[1,:]] = attention_weight

            a_matrix[a_ind[0,:], a_ind[1,:]] = a_weight
            a_matrix[a_ind2[0,:], a_ind2[1,:]] += a_weight2
            a_matrix /= 2 ### num of gat layer
            #import IPython; IPython.embed()
            node_score = F.softmax(self.node_linear(node_out), dim=1)
            node_score_list.append(node_score)

            graph_score = (torch.matmul(a_matrix.T, node_score) + torch.matmul(a_matrix, node_score)).sum(0)\
                              / (len(proposal)*2)
            graph_score = torch.matmul(a_matrix, node_score).sum(0) / len(proposal)

            graph_score_list.append(graph_score)
            import IPython; IPython.embed()
        ref1_logit = self.ref1(x1)
        bbox_pred1 = self.bbox_pred1(x1)
        ref2_logit = self.ref2(x1)
        bbox_pred2 = self.bbox_pred2(x1)
        ref3_logit = self.ref3(x1)
        bbox_pred3 = self.bbox_pred3(x1)

        if not self.training:
            p = 1
            #cls_logit = F.softmax(cls_logit, dim=1)
            # do softmax along ROI for different imgs
            #det_logit_list = det_logit.split([len(p) for p in proposals])
            #final_det_logit = []
            #for det_logit_per_image in det_logit_list:
            #    det_logit_per_image = F.softmax(det_logit_per_image, dim=0)
            #    final_det_logit.append(det_logit_per_image)
            #final_det_logit = torch.cat(final_det_logit, dim=0)
        else:
            #final_det_logit = det_logit
            final_det_logit = []

        ref_logits = [ref1_logit, ref2_logit, ref3_logit]
        bbox_preds = [bbox_pred1, bbox_pred2, bbox_pred3]
        #ref_logits = []
        #bbox_preds = []
        #return cls_logit, final_det_logit, graph_score, node_score, ref_logits, bbox_preds
        return [], [], graph_score_list, node_score_list, ref_logits, bbox_preds

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

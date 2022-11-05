import torch
import collections
import torch.nn as nn
import random
import numpy as np
import scipy.sparse as sp
from torch.nn import functional as F
from itertools import combinations
from wetectron.modeling import registry
from wetectron.modeling.utils import cat
from wetectron.config import cfg
from wetectron.structures.boxlist_ops import boxlist_iou, boxlist_iou_async, boxlist_nms
from wetectron.structures.bounding_box import BoxList
from wetectron.modeling.matcher import Matcher
from wetectron.data.datasets.evaluation.voc.voc_eval import calc_detection_voc_prec_rec

@torch.no_grad()
def to_boxlist(proposal, index):
    boxlist = BoxList(proposal.bbox[index], proposal.size, proposal.mode)
    return boxlist

def cal_iou(proposal, target_index, iou_thres=1e-5):
    iou_index = torch.nonzero(torch.ge(boxlist_iou(proposal, to_boxlist(proposal, target_index.view(-1))), iou_thres).max(dim=1)[0]).view(-1)
    iou_score = boxlist_iou(proposal, to_boxlist(proposal, target_index.view(-1)))[iou_index]
    return iou_index, iou_score

def cal_adj(proposal):
    boxes = BoxList(proposal.bbox, proposal.size, mode=proposal.mode)
    return boxlist_iou(boxes, boxes)

def co_occurrence():
    co_occur = 0
    return co_occur

def normalize_graph(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def temp_softmax(logits, dim=0, T=1):
    e = torch.exp(logits/T)
    return e / e.sum()
    #m = torch.max(logits, dim, keepdim=True)[0]
    #logits = logits/T
    #x_exp = torch.exp(logits-m)
    #x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    #return x_exp/x_exp_sum

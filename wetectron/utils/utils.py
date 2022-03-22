import torch
import collections
import torch.nn as nn
import random
import numpy as np
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

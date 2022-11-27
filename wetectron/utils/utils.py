import torch
import collections
import torch.nn as nn
import random
import numpy as np
import scipy.sparse as sp
import clip

from torch.nn import functional as F
from itertools import combinations
from wetectron.modeling import registry
from wetectron.modeling.utils import cat
from wetectron.config import cfg
from wetectron.structures.boxlist_ops import boxlist_iou, boxlist_iou_async, boxlist_nms
from wetectron.structures.bounding_box import BoxList
from wetectron.modeling.matcher import Matcher
from wetectron.data.datasets.evaluation.voc.voc_eval import calc_detection_voc_prec_rec
from torchvision.transforms.functional import to_pil_image
from torch.distributions import Categorical

CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")

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

def generate_img_label(num_classes, labels, device):
    img_label = torch.zeros(num_classes)
    img_label[labels.long()] = 1
    img_label[0] = 0
    return img_label.to(device)

def txt_embed(model, phrase, list_of_cls, device):
    if len(phrase) == 0:
        token = torch.cat([clip.tokenize(f"{c}") for c in list_of_cls]).to(device)
    else:
        token = torch.cat([clip.tokenize(phrase + f"{c}") for c in list_of_cls]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(token)
    norm_txt = F.normalize(text_embedding, dim=-1).type(torch.float)
    return norm_txt

def img_embed(model, preprocess, imgs, targets, device):
    clip_img_feats = torch.zeros((0, 512), dtype=torch.float, device=device)
    img_label = []
    for img, target in zip(imgs, targets):
        clip_img = preprocess(to_pil_image(img)).unsqueeze(0).to(device)
        img_label.append(int(target.get_field('labels').unique().item() - 1))
        with torch.no_grad():
            image_features = model.encode_image(clip_img)
            clip_img_feats = torch.cat((clip_img_feats, image_features))
    norm_img = F.normalize(clip_img_feats, dim=-1)
    return norm_img, img_label

#def run_clip(img_feats, txt_feats, img_label):
def run_clip(model, preprocess, imgs, phrase, CLASSES, targets, device):

    clip_txt_embed = txt_embed(model, phrase, CLASSES, device)
    clip_img_embed, img_label = img_embed(model, preprocess, imgs, targets, device)

    clip_pred = (100 * clip_img_embed @ clip_txt_embed.T).softmax(dim=-1)
    clip_label = [int(v) for v in clip_pred.argmax(dim=1)]
    clip_entropy = Categorical(probs = clip_pred).entropy()

    '''clip_label_list = [clip_label]
    clip_e_list = [clip_entropy]
    for i, txt_feat in enumerate(txt_feats):
        txt_feat_mask = torch.cat((txt_feats[:i,:], txt_feats[i+1:,:]), dim=0)
        clip_pred = (100 * img_feats @ txt_feat_mask.T).softmax(dim=-1)
        clip_label = [int(v) for v in clip_pred.argmax(dim=1)]
        clip_entropy = Categorical(probs = clip_pred).entropy()

        clip_label_list.append(clip_label)
        clip_e_list.append(clip_entropy)
        #import IPython; IPython.embed()
    '''
    #import IPython; IPython.embed()
    return clip_pred, clip_label, clip_entropy, img_label

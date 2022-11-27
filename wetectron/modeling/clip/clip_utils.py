import torch
import clip

from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image
from torch.distributions import Categorical

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
    '''
    return clip_pred, clip_label, clip_entropy, img_label

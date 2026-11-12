# -*- coding: utf-8 -*-
"""Extract 36 organ-level RADAR visual embeddings (256 dimensions each).

The extractor loads the released visual branch and organ-query heads. It keeps
the original text-embedding path only because the legacy inference flow shares
that structure; zero-shot probabilities are outside this study package's
current scope. The Chinese organ/finding literals below are immutable keys for
the released checkpoint/text-embedding vocabulary and must not be translated.
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from monai import transforms
from monai.data.utils import dense_patch_slices

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RADAR_CODE = os.environ.get("FATTYLIVER_RADAR_CODE", os.path.join(SCRIPT_DIR, "local_model_code", "radar"))
sys.path.insert(0, RADAR_CODE)
from dynamic_network_architectures.vision_branch import VisionBranch

ORGANS = [
    '肾上腺', '主动脉', '竖脊肌', '脑', '锁骨', '大肠', '十二指肠', '食管', '面部', '股骨',
    '胆囊', '臀肌', '心脏', '髋关节', '肱骨', '髂动脉', '髂静脉', '髂腰肌', '下腔静脉', '肾',
    '肝', '肺', '胰腺', '门静脉', '肺动脉', '肋骨', '骶骨', '肩胛骨', '小肠', '脾',
    '胃', '气管', '膀胱', '颈椎', '腰椎', '胸椎'
]

TEST_ITEMS = ['主动脉_主动脉夹层', '主动脉_主动脉瘤', '主动脉_粥样硬化', '主动脉_钙化', '十二指肠_占位', '十二指肠_囊袋状突出影', '十二指肠_憩室', '十二指肠_梗阻', '十二指肠_溃疡', '大肠_克罗恩病', '大肠_大肠（壁）钙化', '大肠_急慢性（结）肠炎', '大肠_浆膜面毛糙', '大肠_溃疡性结肠炎', '大肠_直肠癌', '大肠_积液积气', '大肠_结肠癌', '大肠_肠壁毛糙', '大肠_肠壁水肿', '大肠_肠套叠', '大肠_肠憩室', '大肠_肠梗阻', '大肠_肠穿孔', '大肠_肠道扩张', '大肠_脂肪间隙模糊', '大肠_阑尾炎', '大肠_阑尾粪石', '小肠_克罗恩病', '小肠_套叠', '小肠_扭转', '小肠_梗阻', '小肠_淋巴瘤', '小肠_积气积液', '小肠_系膜指膜炎', '小肠_系膜淋巴结肿大', '小肠_肠壁增厚', '小肠_肠管扩张', '小肠_脂肪瘤', '小肠_间质瘤（胃肠间质瘤-gist）', '小肠_（急慢性）小肠炎', '心脏_心包积液', '心脏_心影（脏）增大', '肋骨_转移瘤（乳腺癌 骨转移）', '肋骨_骨折', '肋骨_骨质破坏', '肝_低密度影', '肝_格林森鞘积液', '肝_比例失调', '肝_波浪状改变', '肝_硬化', '肝_结节状强化', '肝_肝内胆管扩张', '肝_肝内胆管结石', '肝_肝内钙化灶', '肝_肝囊肿', '肝_肝细胞癌', '肝_肝胆管内高密度影', '肝_肝血管瘤', '肝_胆管癌', '肝_脂肪肝', '肝_脓肿', '肝_转移瘤', '肝_边缘不规则', '肺_斑片影', '肺_气胸', '肺_结节', '肺_肺占位', '肺_肺萎陷', '肺_胸腔积液', '肺_膨胀不全', '肺_转移瘤', '肺_钙化灶', '肺_高密度影', '肾_低密度影', '肾_囊肿', '肾_多囊肾', '肾_实质变薄', '肾_无强化囊性灶', '肾_肾动脉瘤', '肾_肾盂扩张', '肾_肾盂癌', '肾_肾盂积水', '肾_肾细胞癌（透明细胞癌）', '肾_肾萎缩', '肾_肾血管平滑肌脂肪瘤', '肾_肾（盂）结石', '肾_高密度影', '肾上腺_增生', '肾上腺_结节', '肾上腺_脂肪瘤', '肾上腺_腺瘤', '肾上腺_转移瘤', '肾上腺_钙化', '胃_壁水肿', '胃_扩张', '胃_胃底静脉曲张', '胃_胃溃疡', '胃_胃癌', '胃_间质瘤（gist）', '胆囊_结石', '胆囊_结节状致密影', '胆囊_胆囊增大', '胆囊_胆囊炎', '胆囊_胆囊癌', '胆囊_胆囊腺肌症', '胆囊_胆管壁增厚', '胆囊_胆管扩张', '胆囊_胆管炎', '胆囊_胆管癌', '胆囊_胆管积气', '胆囊_胆管结石', '胆囊_高密度影', '胆囊_黄色肉芽肿', '胰腺_低密度影', '胰腺_囊肿', '胰腺_围脂肪间隙模糊', '胰腺_肿瘤或胰腺癌', '胰腺_胰周假性囊肿', '胰腺_胰管扩张', '胰腺_胰管结石', '胰腺_胰腺炎', '胰腺_胰腺饱满', '胰腺_萎缩', '脾_低密度灶', '脾_副脾', '脾_囊肿', '脾_梗死', '脾_片状低密度区', '脾_脾大', '脾_脾脏淋巴瘤', '脾_钙化', '膀胱_憩室', '膀胱_结石', '膀胱_膀胱壁毛糙', '膀胱_膀胱炎', '膀胱_膀胱癌', '膀胱_软组织密度影', '门静脉_增宽', '门静脉_栓塞', '门静脉_高压', '食管_增粗迂曲血管影', '食管_管壁增厚', '食管_裂孔疝', '食管_静脉扩张迂曲', '食管_静脉曲张', '骶骨_骨炎']

ROI_SIZE = (96, 256, 384)
OVERLAP = 0.25
SW_BATCH = 1  # Larger cases can exhaust GPU memory with a batch size of 2.


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def build_model(ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['model']
    vision = VisionBranch()
    vis_sd = {k[len('visual_encoder.'):]: v for k, v in sd.items() if k.startswith('visual_encoder.')}
    vision.load_state_dict(vis_sd, strict=True)
    attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=0.1, batch_first=True)
    att_sd = {k[len('attention.'):]: v for k, v in sd.items() if k.startswith('attention.')}
    attention.load_state_dict(att_sd, strict=True)
    vision_projs = nn.ModuleList([nn.Linear(256, 256) for _ in range(len(ORGANS))])
    with torch.no_grad():
        for i in range(len(ORGANS)):
            vision_projs[i].weight.copy_(sd['vision_projs.%d.weight' % i])
            vision_projs[i].bias.copy_(sd['vision_projs.%d.bias' % i])
    query_tokens = sd['query_tokens'].clone()
    temp = sd['temp'].item()
    log('Model assembled: vision=%d keys attention=OK vision_projs=%d query_tokens=%s temp=%.4f'
        % (len(vis_sd), len(vision_projs), tuple(query_tokens.shape), temp))
    return (vision.cuda().eval(), attention.cuda().eval(),
            vision_projs.cuda().eval(), query_tokens.cuda(), temp)


_pad_func = transforms.SpatialPadd(keys=['image'], spatial_size=ROI_SIZE, mode='constant',
                                   constant_values=0, method='end')


def preprocess(image_path):
    res = transforms.LoadImaged(keys=['image'], image_only=False, ensure_channel_first=True)({'image': image_path})
    image = res['image']
    affine = res['image_meta_dict']['affine']
    spacing = (abs(affine[0, 0].item()), abs(affine[1, 1].item()), abs(affine[2, 2].item()))
    _, h, w, d = image.shape
    ref_spacing = (1.0, 1.0, 5.0)
    scale = [spacing[i] / ref_spacing[i] for i in range(3)]
    target_size = [int(h * scale[1]), int(w * scale[0]), int(d * scale[2])]
    trans = transforms.Compose([
        transforms.Resized(spatial_size=target_size, keys=['image'], mode='trilinear'),
        transforms.Transposed(keys=['image'], indices=(0, 3, 2, 1)),
    ])
    image = trans(res)['image']  # [C, D', W', H']
    image[image > 400] = 400
    image[image < -300] = -300
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    roi_coords = np.nonzero(image[0].cpu().numpy())
    min_dhw = torch.from_numpy(np.min(roi_coords, axis=1))
    max_dhw = torch.from_numpy(np.max(roi_coords, axis=1))
    extend_d, extend_hw = 5, 20
    min_dhw = torch.max(min_dhw - torch.tensor([extend_d, extend_hw, extend_hw]), torch.tensor([0, 0, 0]))
    max_dhw = torch.min(max_dhw + torch.tensor([extend_d, extend_hw, extend_hw]),
                        torch.tensor([image.shape[1], image.shape[2], image.shape[3]]))
    cropped = image[:, min_dhw[0]:max_dhw[0], min_dhw[1]:max_dhw[1], min_dhw[2]:max_dhw[2]]
    return _pad_func({'image': cropped})['image'].as_tensor()


def masks_to_boxes_3d(masks):
    d, h, w = masks.shape[-3:]
    z = torch.arange(0, d, dtype=torch.float, device=masks.device)
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    z, y, x = torch.meshgrid(z, y, x, indexing='ij')
    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1).values
    x_min = x_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values
    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1).values
    y_min = y_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values
    z_mask = masks * z.unsqueeze(0)
    z_max = z_mask.flatten(1).max(-1).values
    z_min = z_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values
    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=1)


def center_crop(image, mask, crop_size):
    x_min, y_min, z_min, x_max, y_max, z_max = masks_to_boxes_3d(mask)[0].long()
    crop_d = max(crop_size[0], z_max - z_min)
    crop_h = max(crop_size[1], y_max - y_min)
    crop_w = max(crop_size[2], x_max - x_min)
    cx, cy, cz = (x_min + x_max) // 2, (y_min + y_max) // 2, (z_min + z_max) // 2
    d, h, w = image.shape[-3:]
    x_start = max(0, cx - crop_w // 2); x_end = min(w, x_start + crop_w)
    if x_end - x_start < crop_w: x_start = max(0, x_end - crop_w)
    y_start = max(0, cy - crop_h // 2); y_end = min(h, y_start + crop_h)
    if y_end - y_start < crop_h: y_start = max(0, y_end - crop_h)
    z_start = max(0, cz - crop_d // 2); z_end = min(d, z_start + crop_d)
    if z_end - z_start < crop_d: z_start = max(0, z_end - crop_d)
    return image[..., z_start:z_end, y_start:y_end, x_start:x_end], mask[..., z_start:z_end, y_start:y_end, x_start:x_end]


@torch.inference_mode()
def forward_win(model_parts, images, organ_feat_dict, organ_logits, test_organs, text_feat_dict, skip_organ=None):
    vision, attention, vision_projs, query_tokens, temp = model_parts
    seg_probs, seg, e1, e2, e3, f1, f2, f3 = vision(images, None)
    margin = 2
    masks = seg
    for i, (embed1, embed2, embed3, mask) in enumerate(zip(e1, e2, e3, masks)):
        boundaries = []
        for dd in range(mask.dim()):
            s = [slice(None)] * mask.dim(); e = [slice(None)] * mask.dim()
            s[dd] = slice(None, margin); e[dd] = slice(-margin, None)
            boundaries.append(mask[tuple(s)][mask[tuple(s)] > 0])
            boundaries.append(mask[tuple(e)][mask[tuple(e)] > 0])
        boundaries = torch.cat(boundaries)
        boundary_organs = torch.unique(boundaries[boundaries > 0].flatten())
        if skip_organ is not None:
            boundary_organs = boundary_organs[boundary_organs != skip_organ + 1]
        organ_ids, organ_counts = torch.unique(mask, return_counts=True)
        organ_ids = organ_ids.long()
        organ_counts = organ_counts[organ_ids != 0]
        organ_ids = organ_ids[organ_ids != 0]
        intact_ids = [oid for oid in organ_ids if oid not in boundary_organs]
        if not len(intact_ids):
            continue
        intact_ids = torch.tensor(intact_ids, device=masks.device).long() - 1
        for organ_id in intact_ids:
            organ_name = ORGANS[organ_id.item()]
            if organ_name not in test_organs or organ_name in organ_feat_dict:
                continue
            t1 = f1[i, organ_id, :]; t2 = f2[i, organ_id, :]; t3 = f3[i, organ_id, :]
            query = query_tokens[organ_id].unsqueeze(0).unsqueeze(0)
            key = value = torch.cat([embed1[t1].unsqueeze(0), embed2[t2].unsqueeze(0), embed3[t3].unsqueeze(0)], dim=1)
            updated, _ = attention(query, key, value)
            image_feat = F.normalize(vision_projs[organ_id](updated.squeeze(0)), dim=-1)
            organ_feat_dict[organ_name] = image_feat[0].cpu().numpy()
            for item in organ_logits.keys():
                if item.split('_')[0] != organ_name:
                    continue
                logits = image_feat @ text_feat_dict[item].t().cuda() / temp
                organ_logits[item].append(logits.softmax(-1).cpu().numpy())
    return seg_probs


def _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap):
    out = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            out.append(int(roi_size[i]))
        else:
            iv = int(roi_size[i] * (1 - overlap))
            out.append(iv if iv > 0 else 1)
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir', required=True, help='directory containing NIfTI CT images')
    ap.add_argument('--ckpt', required=True, help='released RADAR checkpoint')
    ap.add_argument('--text_embed', required=True, help='released RADAR text-embedding tensor')
    ap.add_argument('--out_dir', required=True, help='output directory')
    ap.add_argument('--only', default='', help='comma-separated case IDs for debugging')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log('Loading visual branch and organ-query heads ...')
    model_parts = build_model(args.ckpt)
    log('Loading text embeddings ...')
    text_feat_dict = torch.load(args.text_embed, map_location='cpu', weights_only=False)
    miss = [it for it in TEST_ITEMS if it not in text_feat_dict]
    if miss:
        log('Warning: %d findings lack text embeddings: %s' % (len(miss), miss[:3]))
    test_organs = list(set([it.split('_')[0] for it in TEST_ITEMS]))

    files = sorted([f for f in os.listdir(args.img_dir) if f.endswith('.nii.gz')])
    if args.only:
        keep = set(args.only.split(','))
        files = [f for f in files if any(k in f for k in keep)]
    log('Cases queued: %d' % len(files))

    feat_rows, prob_rows = [], []
    t0 = time.time()
    for idx, fn in enumerate(tqdm(files, desc='Cases')):
        tc = time.time()
        image = preprocess(os.path.join(args.img_dir, fn))
        if max(image.shape[1:]) > 1000:
            log('Skipping %s (dimensions %s exceed limit)' % (fn, tuple(image.shape[1:])))
            continue
        image = image[None].cuda()
        image_size = list(image.shape[2:])
        scan_interval = _get_scan_interval(image_size, ROI_SIZE, 3, OVERLAP)
        slices = dense_patch_slices(image_size, ROI_SIZE, scan_interval)
        organ_logits = dict(zip(TEST_ITEMS, [[] for _ in TEST_ITEMS]))
        organ_feat = {}

        full_mask = torch.zeros((1, 37) + tuple(image_size)).cuda()
        count_map = torch.zeros_like(full_mask).cuda()
        for g in range(0, len(slices), SW_BATCH):
            rng = range(g, min(g + SW_BATCH, len(slices)))
            win = [[slice(int(s / len(slices)), int(s / len(slices)) + 1), slice(None)] + list(slices[s % len(slices)]) for s in rng]
            patches = torch.cat([image[w] for w in win]).cuda()
            seg_prob = forward_win(model_parts, patches, organ_feat, organ_logits, test_organs, text_feat_dict)
            interp = F.interpolate(seg_prob, size=patches.shape[2:], mode='trilinear')
            for ii, s in enumerate(rng):
                full_mask[win[ii]] += interp[ii]
                count_map[win[ii]] += 1
        count_map = torch.clamp(count_map, min=1)
        stitched = (full_mask / count_map).argmax(1).unsqueeze(0)

        # Re-run a centre crop for organs not fully captured by any window.
        sq = stitched.squeeze(0).squeeze(0)
        for k, v in organ_logits.items():
            if len(v):
                continue
            organ_name = k.split('_')[0]
            organ_id = ORGANS.index(organ_name)
            if not torch.any(sq == organ_id + 1):
                continue
            patch, pmask = center_crop(image, torch.eq(stitched, organ_id + 1), crop_size=ROI_SIZE)
            pmask = pmask.float(); pmask[pmask == 1] = organ_id + 1
            pad_d = transforms.DivisiblePadd(keys=['image', 'label'], k=32, mode='constant', constant_values=0, method='end')(
                {'image': patch[0], 'label': pmask[0]})
            forward_win(model_parts, pad_d['image'][None].cuda(), organ_feat, organ_logits,
                        test_organs, text_feat_dict, skip_organ=organ_id)

        case_id = fn.replace('.nii.gz', '')
        fr = {'case_id': case_id}
        for organ_name, feat in organ_feat.items():
            for j in range(256):
                fr['%s_%d' % (organ_name, j)] = float(feat[j])
        feat_rows.append(fr)
        pr = {'case_id': case_id}
        for item, probs in organ_logits.items():
            if len(probs):
                pr[item] = float(np.concatenate(probs).mean(0)[1])
        prob_rows.append(pr)
        tqdm.write('[%s] %s complete | windows=%d organs=%d findings=%d elapsed=%.1fs'
                   % (time.strftime('%H:%M:%S'), case_id, len(slices), len(organ_feat),
                      sum(1 for v in organ_logits.values() if v), time.time() - tc))
        del image, full_mask, count_map, stitched
        torch.cuda.empty_cache()

    fdf = pd.DataFrame(feat_rows)
    pdf = pd.DataFrame(prob_rows)
    fpath = os.path.join(args.out_dir, 'radar_organ_features.csv')
    ppath = os.path.join(args.out_dir, 'radar_findings_146.csv')
    fdf.to_csv(fpath, index=False)
    pdf.to_csv(ppath, index=False, encoding='utf-8-sig')
    log('Complete: %d cases | elapsed %.1f min' % (len(fdf), (time.time() - t0) / 60))
    log('Organ features -> %s (%d columns)' % (fpath, fdf.shape[1]))
    log('Finding probabilities -> %s (%d columns)' % (ppath, pdf.shape[1]))
    if '肝_脂肪肝' in pdf.columns:
        log('Hepatic-steatosis probability range: %.4f to %.4f' % (pdf['肝_脂肪肝'].min(), pdf['肝_脂肪肝'].max()))


if __name__ == '__main__':
    main()

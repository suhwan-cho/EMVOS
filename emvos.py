import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


# pre, post processing modules
def aggregate_objects(pred_seg, object_ids):
    bg_seg, _ = torch.stack([seg[:, 0, :, :] for seg in pred_seg.values()], dim=1).min(dim=1)
    bg_seg = torch.stack([1 - bg_seg, bg_seg], dim=1)
    logit = {n: seg[:, 1:, :, :].clamp(1e-7, 1 - 1e-7) / seg[:, 0, :, :].clamp(1e-7, 1 - 1e-7)
             for n, seg in [(-1, bg_seg)] + list(pred_seg.items())}
    logit_sum = torch.cat(list(logit.values()), dim=1).sum(dim=1, keepdim=True)
    aggregated_lst = [logit[n] / logit_sum for n in [-1] + object_ids]
    aggregated_inv_lst = [1 - elem for elem in aggregated_lst]
    aggregated = torch.cat([elem for lst in zip(aggregated_inv_lst, aggregated_lst) for elem in lst], dim=-3)
    mask_tmp = aggregated[:, 1::2, :, :].argmax(dim=-3, keepdim=True)
    pred_mask = torch.zeros_like(mask_tmp)
    for idx, obj_idx in enumerate(object_ids):
        pred_mask[mask_tmp == (idx + 1)] = obj_idx
    return pred_mask, {obj_idx: aggregated[:, 2 * (idx + 1):2 * (idx + 2), :, :] for idx, obj_idx in enumerate(object_ids)}


def get_padding(h, w, div):
    h_pad = (div - h % div) % div
    w_pad = (div - w % div) % div
    padding = [(w_pad + 1) // 2, w_pad // 2, (h_pad + 1) // 2, h_pad // 2]
    return padding


def attach_padding(imgs, given_masks, padding):
    B, L, C, H, W = imgs.size()
    imgs = imgs.view(B * L, C, H, W)
    imgs = F.pad(imgs, padding, mode='reflect')
    _, _, height, width = imgs.size()
    imgs = imgs.view(B, L, C, height, width)
    given_masks = [F.pad(label.float(), padding, mode='reflect').long() if label is not None else None for label in given_masks]
    return imgs, given_masks


def detach_padding(output, padding):
    if isinstance(output, list):
        return [detach_padding(x, padding) for x in output]
    else:
        _, _, _, height, width = output.size()
        return output[:, :, :, padding[2]:height - padding[3], padding[0]:width - padding[1]]


# basic modules
class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('relu', nn.ReLU())
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class DeConv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('deconv', nn.ConvTranspose2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = Conv(c, c, 3, 1, 1)
        self.conv2 = nn.Sequential(ConvRelu(c, c, 1, 1, 0), Conv(c, c, 1, 1, 0))
        self.conv3 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        c = torch.sigmoid(self.conv2(F.adaptive_avg_pool2d(x, output_size=(1, 1))) + self.conv2(F.adaptive_max_pool2d(x, output_size=(1, 1))))
        x = x * c
        s = torch.sigmoid(self.conv3(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)))
        x = x * s
        return x


# encoding module
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = tv.models.densenet121(pretrained=True).features
        self.conv0 = backbone.conv0
        self.norm0 = backbone.norm0
        self.relu0 = backbone.relu0
        self.pool0 = backbone.pool0
        self.denseblock1 = backbone.denseblock1
        self.transition1 = backbone.transition1
        self.denseblock2 = backbone.denseblock2
        self.transition2 = backbone.transition2
        self.denseblock3 = backbone.denseblock3
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img):
        x = (img - self.mean) / self.std
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        x = self.denseblock1(x)
        s4 = x
        x = self.transition1(x)
        x = self.denseblock2(x)
        s8 = x
        x = self.transition2(x)
        x = self.denseblock3(x)
        s16 = x
        return {'s16': s16, 's8': s8, 's4': s4}


# matching module
class Matcher(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = Conv(in_c, out_c, 1, 1, 0)
        self.conv2 = Conv(in_c, out_c, 1, 1, 0)

    def get_norm_key(self, x):
        key = self.conv1(x)
        norm_key = key / key.norm(dim=1, keepdim=True)
        return norm_key

    def get_key(self, x):
        key = self.conv2(x)
        return key

    def forward(self, init_cossim, prev_cossim, init_softsim, prev_softsim, state):
        B, _, H, W = init_cossim.size()

        # surjective global matching
        score = init_cossim * state['init_seg_16'][:, 0].view(B, H * W, 1, 1)
        bg_score = torch.max(score, dim=1, keepdim=True)[0]
        score = init_cossim * state['init_seg_16'][:, 1].view(B, H * W, 1, 1)
        fg_score = torch.max(score, dim=1, keepdim=True)[0]
        global_score = torch.cat([bg_score, fg_score], dim=1)

        # surjective local matching
        score = prev_cossim * state['prev_seg_16'][:, 0].view(B, H * W, 1, 1)
        bg_score = torch.max(score, dim=1, keepdim=True)[0]
        score = prev_cossim * state['prev_seg_16'][:, 1].view(B, H * W, 1, 1)
        fg_score = torch.max(score, dim=1, keepdim=True)[0]
        local_score = torch.cat([bg_score, fg_score], dim=1)

        # equalized global matching
        score = init_softsim * state['init_seg_16'][:, 0].view(B, H * W, 1, 1)
        bg_score = torch.max(score, dim=1, keepdim=True)[0]
        score = init_softsim * state['init_seg_16'][:, 1].view(B, H * W, 1, 1)
        fg_score = torch.max(score, dim=1, keepdim=True)[0]
        eq_global_score = torch.cat([bg_score, fg_score], dim=1)

        # equalized local matching
        score = prev_softsim * state['prev_seg_16'][:, 0].view(B, H * W, 1, 1)
        bg_score = torch.max(score, dim=1, keepdim=True)[0]
        score = prev_softsim * state['prev_seg_16'][:, 1].view(B, H * W, 1, 1)
        fg_score = torch.max(score, dim=1, keepdim=True)[0]
        eq_local_score = torch.cat([bg_score, fg_score], dim=1)

        # collect matching scores
        matching_score = torch.cat([global_score, local_score, eq_global_score, eq_local_score], dim=1)
        return matching_score


# decoding module
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvRelu(1024, 256, 1, 1, 0)
        self.blend1 = ConvRelu(256 + 8 + 2, 256, 3, 1, 1)
        self.cbam1 = CBAM(256)
        self.deconv1 = DeConv(256, 2, 4, 2, 1)
        self.conv2 = ConvRelu(512, 256, 1, 1, 0)
        self.blend2 = ConvRelu(256 + 2, 256, 3, 1, 1)
        self.cbam2 = CBAM(256)
        self.deconv2 = DeConv(256, 2, 4, 2, 1)
        self.conv3 = ConvRelu(256, 256, 1, 1, 0)
        self.blend3 = ConvRelu(256 + 2, 256, 3, 1, 1)
        self.cbam3 = CBAM(256)
        self.deconv3 = DeConv(256, 2, 6, 4, 1)

    def forward(self, feats, matching_score, prev_seg_16):
        x = torch.cat([self.conv1(feats['s16']), matching_score, prev_seg_16], dim=1)
        s8 = self.deconv1(self.cbam1(self.blend1(x)))
        x = torch.cat([self.conv2(feats['s8']), s8], dim=1)
        s4 = self.deconv2(self.cbam2(self.blend2(x)))
        x = torch.cat([self.conv3(feats['s4']), s4], dim=1)
        x = self.deconv3(self.cbam3(self.blend3(x)))
        return x


# VOS model
class VOS(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.matcher = Matcher(1024, 512)
        self.decoder = Decoder()

    def get_init_state(self, norm_key, key, init_seg):
        state = {}
        state['init_norm_key'] = norm_key
        state['prev_norm_key'] = norm_key
        state['init_key'] = key
        state['prev_key'] = key
        state['init_seg_16'] = F.avg_pool2d(init_seg, 16)
        state['prev_seg_16'] = F.avg_pool2d(init_seg, 16)
        return state

    def update(self, norm_key, key, pred_seg, full_state, object_ids):
        for k in object_ids:
            full_state[k]['prev_norm_key'] = norm_key
            full_state[k]['prev_key'] = key
            full_state[k]['prev_seg_16'] = F.avg_pool2d(pred_seg[k], 16)
        return full_state

    def forward(self, feats, norm_key, key, full_state, object_ids):
        B, _, H, W = norm_key.size()
        final_score = {}
        for k in object_ids:
            init_norm_key = full_state[k]['init_norm_key'].view(B, -1, H * W).transpose(1, 2)
            prev_norm_key = full_state[k]['prev_norm_key'].view(B, -1, H * W).transpose(1, 2)
            init_cossim = (torch.bmm(init_norm_key, norm_key.view(B, -1, H * W)).view(B, H * W, H, W) + 1) / 2
            prev_cossim = (torch.bmm(prev_norm_key, norm_key.view(B, -1, H * W)).view(B, H * W, H, W) + 1) / 2
            init_key = full_state[k]['init_key'].view(B, -1, H * W).transpose(1, 2)
            prev_key = full_state[k]['prev_key'].view(B, -1, H * W).transpose(1, 2)
            init_softsim = torch.softmax(torch.bmm(init_key, key.view(B, -1, H * W)), dim=2).view(B, H * W, H, W)
            prev_softsim = torch.softmax(torch.bmm(prev_key, key.view(B, -1, H * W)), dim=2).view(B, H * W, H, W)
            matching_score = self.matcher(init_cossim, prev_cossim, init_softsim, prev_softsim, full_state[k])
            final_score[k] = self.decoder(feats, matching_score, full_state[k]['prev_seg_16'])
        return final_score


# EMVOS model
class EMVOS(nn.Module):
    def __init__(self):
        super().__init__()
        self.vos = VOS()

    def forward(self, imgs, given_masks, val_frame_ids):

        # basic setting
        B, L, _, H, W = imgs.size()
        padding = get_padding(H, W, 16)
        if B != 1:
            given_masks = [given_masks[:, 0]] + (L - 1) * [None]
        if tuple(padding) != (0, 0, 0, 0):
            imgs, given_masks = attach_padding(imgs, given_masks, padding)
        _, _, _, H, W = imgs.size()

        # initial frame
        init_mask = given_masks[0]
        object_ids = init_mask.unique().tolist()
        if 0 in object_ids:
            object_ids.remove(0)
        score_lst = []
        mask_lst = [given_masks[0]]

        # extract features
        with torch.no_grad():
            feats = self.vos.encoder(imgs[:, 0])
        norm_key = self.vos.matcher.get_norm_key(feats['s16'])
        key = self.vos.matcher.get_key(feats['s16'])

        # create state for each object
        state = {}
        for k in object_ids:
            init_seg = torch.cat([init_mask != k, init_mask == k], dim=1).float()
            state[k] = self.vos.get_init_state(norm_key, key, init_seg)

        # subsequent frames
        for i in range(1, L):

            # query frame prediction
            with torch.no_grad():
                feats = self.vos.encoder(imgs[:, i])
            norm_key = self.vos.matcher.get_norm_key(feats['s16'])
            key = self.vos.matcher.get_key(feats['s16'])
            final_score = self.vos(feats, norm_key, key, state, object_ids)
            pred_seg = {k: F.softmax(final_score[k], dim=1) for k in object_ids}

            # detect new object
            if given_masks[i] is not None:
                new_object_ids = given_masks[i].unique().tolist()
                if 0 in new_object_ids:
                    new_object_ids.remove(0)
                for new_k in new_object_ids:
                    init_seg = torch.cat([given_masks[i] != new_k, given_masks[i] == new_k], dim=1).float()
                    state[new_k] = self.vos.get_init_state(norm_key, key, init_seg)
                    pred_seg[new_k] = torch.cat([given_masks[i] != new_k, given_masks[i] == new_k], dim=1).float()
                object_ids = object_ids + new_object_ids

            # aggregate objects
            if B == 1:
                pred_mask, pred_seg = aggregate_objects(pred_seg, object_ids)

            # update state
            if i < L - 1:
                state = self.vos.update(norm_key, key, pred_seg, state, object_ids)

            # generate soft scores
            if B != 1:
                score_lst.append(final_score[1])

            # generate hard masks
            if B == 1:
                if given_masks[i] is not None:
                    pred_mask[given_masks[i] != 0] = 0
                    mask_lst.append(pred_mask + given_masks[i])
                else:
                    if val_frame_ids is not None:
                        if val_frame_ids[0] + i in val_frame_ids:
                            mask_lst.append(pred_mask)
                    else:
                        mask_lst.append(pred_mask)

        # store output
        output = {}
        if B != 1:
            output['scores'] = torch.stack(score_lst, dim=1)
            output['scores'] = detach_padding(output['scores'], padding)
        if B == 1:
            output['masks'] = torch.stack(mask_lst, dim=1)
            output['masks'] = detach_padding(output['masks'], padding)
        return output

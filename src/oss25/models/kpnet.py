import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class KPNet(nn.Module):
    def __init__(self, encoder_name: str = 'convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384',
                 image_size: int = 384, heatmap_size: int = 96, num_keypoints: int = 27,
                 in_chans: int = 3):
        super().__init__()
        def _parse_hw(v):
            if isinstance(v, (tuple, list)):
                return int(v[0]), int(v[1])
            n = int(v)
            return n, n
        self.image_h, self.image_w = _parse_hw(image_size)
        self.heatmap_h, self.heatmap_w = _parse_hw(heatmap_size)
        self.image_size = (self.image_h, self.image_w)
        self.heatmap_size = (self.heatmap_h, self.heatmap_w)
        self.num_keypoints = num_keypoints
        self.in_chans = in_chans
        self.num_seg_classes = 7  # background + 6 classes

        # Encoder from timm, return feature map (no pooling)
        # Use features_only to avoid classifier head mismatch and get spatial feature maps
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=False,
            features_only=True,
            out_indices=(1, 2, 3),  # multi-scale for FPN
            in_chans=in_chans,
        )

        # Infer channels of last feature
        with torch.no_grad():
            x = torch.zeros(1, self.in_chans, self.image_h, self.image_w)
            feats = self.encoder(x)
            if isinstance(feats, (list, tuple)):
                f1, f2, f3 = feats  # strides ~8,16,32
            else:
                # fallback, treat as single feature map
                f1 = f2 = f3 = feats
            c1, c2, c3 = f1.shape[1], f2.shape[1], f3.shape[1]
            h1, w1 = f1.shape[-2], f1.shape[-1]

        # Lateral 1x1 conv to build FPN
        self.lat1 = nn.Conv2d(c1, 256, kernel_size=1)
        self.lat2 = nn.Conv2d(c2, 256, kernel_size=1)
        self.lat3 = nn.Conv2d(c3, 256, kernel_size=1)
        self.smooth1 = ConvBlock(256, 256)
        self.smooth2 = ConvBlock(256, 256)
        self.smooth3 = ConvBlock(256, 256)

        # Simple upsampling decoder from the finest FPN level to near target heatmap resolution.
        up_layers = []
        cur_ch = 256
        cur_h, cur_w = h1, w1
        # progressively upsample but avoid overshooting by more than x2
        while (cur_h * 2) <= self.heatmap_h and (cur_w * 2) <= self.heatmap_w:
            out_ch = max(256, cur_ch // 2)
            up_layers.append(ConvBlock(cur_ch, out_ch))
            up_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            cur_ch = out_ch
            cur_h *= 2
            cur_w *= 2
            if len(up_layers) > 12:  # safety
                break
        self.decoder = nn.Sequential(*up_layers)
        dec_out_ch = cur_ch
        self.decoder_out_hw = (cur_h, cur_w)

        # Heads
        self.hm_head = nn.Conv2d(dec_out_ch, num_keypoints, kernel_size=1)
        # 3-class visibility logits per kp -> output shape (B, num_keypoints*3)
        self.vis_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # B,C,1,1
            nn.Conv2d(dec_out_ch, 512, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.vis_classifier = nn.Linear(512, num_keypoints * 3)
        # Auxiliary segmentation head (7 classes: bg + 6 tools) on the same decoded feature
        self.seg_head = nn.Conv2d(dec_out_ch, self.num_seg_classes, kernel_size=1)

    def forward(self, x, return_seg: bool = False):
        feats = self.encoder(x)
        if isinstance(feats, (list, tuple)) and len(feats) >= 3:
            f1, f2, f3 = feats  # low->high
        else:
            f1 = f2 = f3 = feats
        # FPN top-down
        p3 = self.lat3(f3)
        p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode='nearest')
        p1 = self.lat1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode='nearest')
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)
        y = self.decoder(p1)
        heatmaps = self.hm_head(y)
        # resize to exact heatmap size if needed
        if heatmaps.shape[-2] != self.heatmap_h or heatmaps.shape[-1] != self.heatmap_w:
            heatmaps = F.interpolate(heatmaps, size=(self.heatmap_h, self.heatmap_w), mode='bilinear', align_corners=False)
        g = self.vis_proj(y).flatten(1)
        vis_logits = self.vis_classifier(g).view(-1, self.num_keypoints, 3)
        if return_seg:
            seg_logits = self.seg_head(y)
            if seg_logits.shape[-2] != self.heatmap_h or seg_logits.shape[-1] != self.heatmap_w:
                seg_logits = F.interpolate(seg_logits, size=(self.heatmap_h, self.heatmap_w), mode='bilinear', align_corners=False)
            return heatmaps, vis_logits, seg_logits
        return heatmaps, vis_logits

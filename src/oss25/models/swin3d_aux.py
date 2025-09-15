import torch
import torch.nn as nn
from torchvision.models.video import swin3d_b, Swin3D_B_Weights


class Swin3DWithSegAux(nn.Module):
    """
    Swin3D-B backbone with two heads:
      - main head: classification (TaskA) or regression (TaskB)
      - seg aux head: Conv3d over features to predict 7-class pseudo masks (0..6)

    Notes:
    - Input: BxCxTxHxW
    - Features shape (for 224x224, T=96): BxT' x 7 x 7 x 1024 (T' ~ 48)
    - seg_head consumes permuted features Bx1024xT'x7x7 and outputs Bx7xT'x7x7
    """

    def __init__(self, task: str, num_outputs: int, weights: str = 'KINETICS400_IMAGENET22K_V1'):
        super().__init__()
        if weights:
            w = getattr(Swin3D_B_Weights, weights)
            self.backbone = swin3d_b(weights=w)
        else:
            self.backbone = swin3d_b(weights=None)

        # Replace classification head with identity to expose 1024-dim features after avgpool
        # We will forward to get pooled features and then our own head
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.task = task  # 'A' or 'B'
        # For TaskA we now use regression (scalar GRS), for TaskB 8-dim regression
        self.main_head = nn.Linear(in_features, num_outputs)

        # Auxiliary segmentation head over features (before final norm+avgpool)
        # We tap into self.backbone.features output by hooking in forward
        self.seg_head = nn.Conv3d(1024, 7, kernel_size=1)

        # Auxiliary task heads (regularization)
        # TIME binary classification (pre/post)
        self.time_head = nn.Linear(in_features, 2)
        # GROUP 3-class classification (E-LEARNING, HMD-BASED, TUTOR-LED)
        self.group_head = nn.Linear(in_features, 3)
        # SUTURES scalar regression
        self.sutures_head = nn.Linear(in_features, 1)

    @staticmethod
    def _temporal_out_len(T_in: int) -> int:
        # Conv3d in PatchEmbed3d uses kernel_size=(2,4,4), stride=(2,4,4), padding=0
        # T_out = floor((T_in - 2)/2 + 1)
        return (T_in - 2) // 2 + 1

    def forward(self, x: torch.Tensor):
        """
        Returns dict with:
          - logits or preds: main output
          - seg_logits: Bx7xT'x7x7 (for aux loss)
          - feats_meta: dict with T', H', W'
        """
        b = self.backbone

        # Patch embedding
        x = b.patch_embed(x)
        # x: B x T'0 x H'0 x W'0 x C
        x = b.pos_drop(x)
        x = b.features(x)
        # x now: B x T' x H' x W' x C(=1024)
        features = x
        x = b.norm(x)
        # move channel to 2nd dim for pooling
        x = x.permute((0, 4, 1, 2, 3))
        pooled = b.avgpool(x)  # B x 1024 x 1 x 1 x 1
        pooled = torch.flatten(pooled, 1)

        # main head
        main_out = self.main_head(pooled)
        # If TaskB with 8-dim head, map to [1,5] via 1 + 4*sigmoid
        if self.task == 'B' and main_out.dim() == 2 and main_out.shape[1] == 8:
            main_out = 1.0 + 4.0 * torch.sigmoid(main_out)

        # aux heads (from same pooled feature)
        time_logits = self.time_head(pooled)
        group_logits = self.group_head(pooled)
        sutures_pred = self.sutures_head(pooled).squeeze(-1)

        # seg aux head from features: BxTxHxWxC -> BxCxTxHxW
        seg_x = features.permute(0, 4, 1, 2, 3)
        seg_logits = self.seg_head(seg_x)

        return {
            'main': main_out,
            'seg_logits': seg_logits,
            'time_logits': time_logits,
            'group_logits': group_logits,
            'sutures_pred': sutures_pred,
            'feats_meta': {
                'T_out': seg_logits.shape[2],
                'H_out': seg_logits.shape[3],
                'W_out': seg_logits.shape[4],
            }
        }

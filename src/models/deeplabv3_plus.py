import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import ResNetBackbone
from src.models.aspp import ASPP
from src.models.decoder import DeepLabV3PlusDecoder
from src.models.hybrid_context import HybridContextNeck


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_pretrained: bool = True,
        backbone_name: str = "rsnet-50",
        output_stride: int = 16,
        aspp_out_channels: int = 256,
        decoder_channels: int = 256,
        aspp_dropout: float = 0.1,
        decoder_dropout: float = 0.2,
        segmentation_head: str = "aspp",
        hybrid_variant: str = "large",
        hybrid_use_strip: bool = False,
        hybrid_strip_kernel: int = 11,
        hybrid_mid_kernel: int = 7,
        hybrid_large_kernel: int = 15,
        hybrid_gate_reduction: int = 16,
        hybrid_residual_channels: int = 128,
        hybrid_residual_init: float = 0.02,
        hybrid_dropout: float = 0.05,
        decoder_upsample_mode: str = "learnable",
    ):
        super().__init__()
        self.segmentation_head = str(segmentation_head).lower()

        self.backbone = ResNetBackbone(
            pretrained=backbone_pretrained,
            output_stride=output_stride,
            backbone_name=backbone_name,
        )

        if output_stride == 16:
            rates = (6, 12, 18)
        elif output_stride == 8:
            rates = (12, 24, 36)
        else:
            raise ValueError("output_stride must be 8 or 16")

        if self.segmentation_head == "aspp":
            self.aspp = ASPP(
                in_channels=self.backbone.out_channels,
                out_channels=aspp_out_channels,
                atrous_rates=rates,
                dropout=aspp_dropout,
            )
            self.hybrid_neck = None
        elif self.segmentation_head == "hybrid":
            self.aspp = None
            self.hybrid_neck = HybridContextNeck(
                in_channels=self.backbone.out_channels,
                out_channels=aspp_out_channels,
                atrous_rates=rates,
                variant=hybrid_variant,
                use_strip=hybrid_use_strip,
                strip_kernel=hybrid_strip_kernel,
                mid_kernel=hybrid_mid_kernel,
                large_kernel=hybrid_large_kernel,
                gate_reduction=hybrid_gate_reduction,
                residual_channels=hybrid_residual_channels,
                residual_init=hybrid_residual_init,
                dropout=hybrid_dropout,
            )
        else:
            raise ValueError(
                f"Unsupported segmentation_head: {segmentation_head}. Use 'aspp' or 'hybrid'."
            )

        self.decoder = DeepLabV3PlusDecoder(
            low_level_in_channels=self.backbone.low_level_channels,
            aspp_out_channels=aspp_out_channels,
            decoder_channels=decoder_channels,
            dropout=decoder_dropout,
            upsample_mode=decoder_upsample_mode,
        )
        self.distill_context_channels = int(aspp_out_channels)
        self.distill_decoder_channels = int(decoder_channels)

        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        self.aux_classifier = nn.Conv2d(self.backbone.out_channels, num_classes, kernel_size=1)

        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.0)

        nn.init.normal_(self.aux_classifier.weight, mean=0.0, std=0.01)
        if self.aux_classifier.bias is not None:
            nn.init.constant_(self.aux_classifier.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        return_preupsample: bool = False,
        return_features: bool = False,
    ):
        input_size = x.shape[-2:]

        low_level, _, high_level = self.backbone(x)
        aux_logits = self.aux_classifier(high_level)
        aux_logits_preupsample = aux_logits

        if self.segmentation_head == "aspp":
            head_feat = self.aspp(high_level)
            context_base = head_feat
            context_residual = torch.zeros_like(head_feat)
        else:
            if return_features:
                head_feat, context_meta = self.hybrid_neck(high_level, return_intermediates=True)
                context_base = context_meta["context_base"]
                context_residual = context_meta["context_residual"]
            else:
                head_feat = self.hybrid_neck(high_level)
                context_base = head_feat
                context_residual = torch.zeros_like(head_feat)

        dec_feat = self.decoder(low_level, head_feat)
        logits = self.classifier(dec_feat)
        logits_preupsample = logits
        logits = F.interpolate(logits_preupsample, size=input_size, mode="bilinear", align_corners=False)
        features = {
            "context": head_feat,
            "context_base": context_base,
            "context_residual": context_residual,
            "decoder": dec_feat,
            "high_level": high_level,
            "low_level": low_level,
        }

        if not return_aux:
            if return_preupsample:
                if return_features:
                    return logits, logits_preupsample, features
                return logits, logits_preupsample
            if return_features:
                return logits, features
            return logits

        aux_logits = F.interpolate(aux_logits_preupsample, size=input_size, mode="bilinear", align_corners=False)
        if return_preupsample:
            if return_features:
                return logits, aux_logits, logits_preupsample, aux_logits_preupsample, features
            return logits, aux_logits, logits_preupsample, aux_logits_preupsample
        if return_features:
            return logits, aux_logits, features
        return logits, aux_logits

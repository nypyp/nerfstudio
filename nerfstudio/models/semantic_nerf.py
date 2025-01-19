
from dataclasses import dataclass, field
from typing import  Any, Dict, Type, Tuple, Literal

import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.fields.semantic_nerf_field import SemanticNerfField
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.model_components.losses import MSELoss, scale_gradients_by_distance_squared
from nerfstudio.model_components.ray_samplers import UniformSampler, PDFSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer, 
    DepthRenderer, 
    RGBRenderer,
    SemanticRenderer,
    UncertaintyRenderer
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import misc, colormaps

@dataclass
class SemanticNerfModelConfig(ModelConfig):
    """Semantic Nerf model configuration

    Args:
        Model (_type_): _description_
    """
    _target: Type = field(default_factory=lambda: SemanticNerfModel)
    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation"""
    num_importance_samples: int = 128
    """Number of samples in fine field evaluation"""
    
    enable_temporal_distortion: bool = False
    """Specifies whether or not to include ray warping based on time"""
    temporal_distortion_params: Dict[str, Any] = to_immutable_dict({"kind": TemporalDistortionKind.DNERF})
    """Parameters to instantiate temporal distortion with"""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """Whether to randomize the background color."""
    
    semantic_loss_weight: float = 0.004


class SemanticNerfModel(Model):
    """Semantic Nerf model

    Args:
        Model (_type_): _description_
    """
    config: SemanticNerfModelConfig
    
    def __init__(
        self,
        config: SemanticNerfModelConfig,
        metadata: Dict,
        **kwargs,
    ) -> None:
        assert "semantics" in metadata.keys() and isinstance(metadata["semantics"], Semantics)
        self.semantics = metadata["semantics"]
        
        self.field_coarse = None
        self.field_fine = None
        self.temporal_distortion = None
        
        super().__init__(
            config=config,
            **kwargs,
        )
        self.colormap = self.semantics.colors.clone().detach().to(self.device)
    
    def populate_modules(self):
        """Set the fields and moduls"""
        super().populate_modules()
        
        # fields
        position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=10,
            min_freq_exp=0.0,
            max_freq_exp=8.0,
            include_input=True
        )
        
        direction_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=4,
            min_freq_exp=0.0,
            max_freq_exp=4.0,
            include_input=True
        )
        
        self.field_coarse = SemanticNerfField(
            num_semantic_classes=len(self.semantics.classes),
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )
        
        self.field_fine = SemanticNerfField(
            num_semantic_classes=len(self.semantics.classes),
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )
        
        #samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)
        
        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_uncertainty = UncertaintyRenderer()
        self.renderer_semantic = SemanticRenderer()
        
        # losses
        self.rgb_loss = MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean")
        
        # metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        
        if getattr(self.config, "enable_temporal_distortion", False):
            params = self.config.temporal_distortion_params
            kind = params.pop("kind")
            self.temporal_distortion = kind.to_temporal_distortion(params)
            
    def get_param_groups(self) -> Dict[str, torch.List[Parameter]]:
        param_groups = {}
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        if self.temporal_distortion is not None:
            param_groups["temporal_distortion"] = list(self.temporal_distortion.parameters())
        return param_groups
    
    def get_outputs(self, ray_bundle: RayBundle):
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")
        
        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_uniform.times is not None:
                offsets = self.temporal_distortion(
                    ray_samples_uniform.frustums.get_positions(),
                    ray_samples_uniform.times
                )
            ray_samples_uniform.frustums.set_offets(offsets)
            
        #coarse field：
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        if self.config.use_gradient_scaling:
            field_outputs_coarse = scale_gradients_by_distance_squared(field_outputs_coarse, ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)
        semantics_coarse = self.renderer_semantic(
            field_outputs_coarse[FieldHeadNames.SEMANTICS],
            weights=weights_coarse,
        )
        
        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_pdf.times is not None:
                offsets = self.temporal_distortion(ray_samples_pdf.frustums.get_positions(), ray_samples_pdf.times)
            ray_samples_pdf.frustums.set_offsets(offsets)
            
        # fine field
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        if self.config.use_gradient_scaling:
            field_outputs_fine = scale_gradients_by_distance_squared(field_outputs_fine, ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)
        semantics_fine = self.renderer_semantic(
            field_outputs_fine[FieldHeadNames.SEMANTICS],
            weights=weights_fine,
        )
        
        # semantic colormaps outputs[semantics_colormap] : [4096, 3]
        semantic_labels = torch.argmax(torch.nn.functional.softmax(semantics_fine, dim=-1), dim=-1)
        semantics_colormap = self.colormap.to(self.device)[semantic_labels]
        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "semantics_coarse": semantics_coarse,
            "semantics_fine": semantics_fine,
            "semantics_colormap": semantics_colormap
        }
        return outputs
        
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the lossese
        device = outputs["rgb_coarse"].device
        image = batch["image"].to(device)
        coarse_pred, coarse_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_coarse"],
            pred_accumulation=outputs["accumulation_coarse"],
            gt_image=image,
        )
        fine_pred, fine_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )
        
        # rgb loss
        rgb_loss_coarse = self.rgb_loss(coarse_image, coarse_pred)
        rgb_loss_fine = self.rgb_loss(fine_image, fine_pred)
        
        # semantic loss
        semantic_loss_coarse = self.config.semantic_loss_weight * self.cross_entropy_loss(
            outputs["semantics_coarse"],
            batch["semantics"][..., 0].long().to(device)
        )
        semantic_loss_fine = self.config.semantic_loss_weight * self.cross_entropy_loss(
            outputs["semantics_fine"],
            batch["semantics"][..., 0].long().to(device)
        )
        
        loss_dict = {
            "rgb_loss_coarse": rgb_loss_coarse,
            "rgb_loss_fine": rgb_loss_fine,
            "semantic_loss_coarse": semantic_loss_coarse,
            "semantic_loss_fine": semantic_loss_fine,
        }
        return loss_dict
    
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb_coarse"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
        assert self.config.collider_params is not None
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"]
        )
        
        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)
        
        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]
        
        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)
        assert isinstance(fine_ssim, torch.Tensor)
        
        # semantics
        semantic_lables = torch.argmax(torch.nn.functional.softmax(outputs["semantics_fine"], dim=-1), dim=-1)
        semantics_colormap = self.colormap.to(self.device)[semantic_lables]
        
        
        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr),
            "fine_psnr": float(fine_psnr),
            "fine_ssim": float(fine_ssim),
            "fine_lpips": float(fine_lpips),
        }
        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "semantic_colormap": semantics_colormap,
        }
        return metrics_dict, images_dict
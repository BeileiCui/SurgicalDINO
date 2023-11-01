import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# from mmcv.cnn import ConvModule
from transformers.modeling_outputs import DepthEstimatorOutput

import sys

REPO_PATH = "/mnt/data-hdd2/Beilei/Repository/dinov2" # Specify a local path to the repository (or use installed package instead)
sys.path.append(REPO_PATH)

from dinov2.eval.depth.models.losses.gradientloss import GradientLoss
from dinov2.eval.depth.models.losses.sigloss import SigLoss
from dinov2.eval.depth.ops import resize
from dinov2.eval.depth.models.decode_heads.dpt_head import ReassembleBlocks, FeatureFusionBlock, HeadDepth

from torch import Tensor
from torch.nn.parameter import Parameter

class decode_head_linear(torch.nn.Module):

    def __init__(self, input_transform="resize_concat", image_shape=(224,224), in_index=(0, 1, 2, 3), upsample=4, min_depth=0.001, max_depth=150, classify=True,
        n_bins=256, bins_strategy='UD', norm_strategy='linear', in_channels=[768,768,768,768], channels=6144, align_corners=False, scale_up=False,):
        super().__init__()
        self.input_transform = input_transform
        self.image_shape = image_shape
        self.in_index = in_index
        self.upsample = upsample
        self.classify = classify
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.bins_strategy = bins_strategy
        self.norm_strategy = norm_strategy
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        self.scale_up = scale_up
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.bn = nn.SyncBatchNorm(self.in_channels)
        if self.classify:
            self.conv_depth = nn.Conv2d(self.channels, self.n_bins, kernel_size=1, padding=0, stride=1)
        else:
            self.conv_depth = nn.Conv2d(self.channels, 1, kernel_size=1, padding=0, stride=1)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if "concat" in self.input_transform:
            inputs = [inputs[i] for i in self.in_index]
            if "resize" in self.input_transform:
                inputs = [
                    resize(
                        input=x,
                        size=[s * self.upsample for s in inputs[0].shape[2:]],
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    for x in inputs
                ]
            inputs = torch.cat(inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _forward_feature(self, inputs, img_metas=None, **kwargs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # accept lists (for cls token)
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            if len(x) == 2:
                x, cls_token = x[0], x[1]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                inputs[i] = torch.cat((x, cls_token), 1)
            else:
                x = x[0]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                inputs[i] = x
        x = self._transform_inputs(inputs)
        # feats = self.bn(x)
        return x
    
    def depth_pred(self, feat):
        """Prediction each pixel."""
        if self.classify:
            logit = self.conv_depth(feat)

            if self.bins_strategy == "UD":
                bins = torch.linspace(self.min_depth, self.max_depth, self.n_bins, device=feat.device)
            elif self.bins_strategy == "SID":
                bins = torch.logspace(self.min_depth, self.max_depth, self.n_bins, device=feat.device)

            # following Adabins, default linear
            if self.norm_strategy == "linear":
                logit = torch.relu(logit)
                eps = 0.1
                logit = logit + eps
                logit = logit / logit.sum(dim=1, keepdim=True)
            elif self.norm_strategy == "softmax":
                logit = torch.softmax(logit, dim=1)
            elif self.norm_strategy == "sigmoid":
                logit = torch.sigmoid(logit)
                logit = logit / logit.sum(dim=1, keepdim=True)

            output = torch.einsum("ikmn,k->imn", [logit, bins]).unsqueeze(dim=1)
        return output
    
    def forward(self, inputs, **kwargs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.depth_pred(output)
        output = torch.nn.functional.interpolate(output, size=self.image_shape, mode="bilinear", align_corners=self.align_corners)
        # print(output.shape)
        return output
    

# class decode_head_dpt(torch.nn.Module):
#     """Vision Transformers for Dense Prediction.
#     This head is implemented of `DPT <https://arxiv.org/abs/2103.13413>`_.
#     Args:
#         embed_dims (int): The embed dimension of the ViT backbone.
#             Default: 768.
#         post_process_channels (List): Out channels of post process conv
#             layers. Default: [96, 192, 384, 768].
#         readout_type (str): Type of readout operation. Default: 'ignore'.
#         patch_size (int): The patch size. Default: 16.
#         expand_channels (bool): Whether expand the channels in post process
#             block. Default: False.
#     """

#     def __init__(
#         self,
#         norm_cfg=None,
#         act_cfg=dict(type="ReLU"),
#         min_depth=0.001,
#         max_depth=150, 
#         classify=False,
#         scale_up=False,
#         in_channels=[768,768,768,768], 
#         channels=256,
#         embed_dims=768,
#         post_process_channels=[96, 192, 384, 768],
#         readout_type="ignore",
#         patch_size=16,
#         expand_channels=False,
#         **kwargs
#     ):
#         super(decode_head_dpt, self).__init__(**kwargs)

#         self.norm_cfg = norm_cfg
#         self.act_cfg = act_cfg
#         self.min_depth = min_depth
#         self.max_depth = max_depth
#         self.classify = classify
#         self.scale_up = scale_up
#         self.in_channels = in_channels
#         self.channels = channels
#         self.expand_channels = expand_channels
        
#         self.reassemble_blocks = ReassembleBlocks(embed_dims, post_process_channels, readout_type, patch_size)

#         self.post_process_channels = [
#             channel * math.pow(2, i) if expand_channels else channel for i, channel in enumerate(post_process_channels)
#         ]
#         self.convs = nn.ModuleList()
#         for channel in self.post_process_channels:
#             self.convs.append(ConvModule(channel, self.channels, kernel_size=3, padding=1, act_cfg=None, bias=False))
#         self.fusion_blocks = nn.ModuleList()
#         for _ in range(len(self.convs)):
#             self.fusion_blocks.append(FeatureFusionBlock(self.channels, self.act_cfg, self.norm_cfg))
#         self.fusion_blocks[0].res_conv_unit1 = None
#         self.project = ConvModule(self.channels, self.channels, kernel_size=3, padding=1, norm_cfg=self.norm_cfg)
#         self.num_fusion_blocks = len(self.fusion_blocks)
#         self.num_reassemble_blocks = len(self.reassemble_blocks.resize_layers)
#         self.num_post_process_channels = len(self.post_process_channels)
#         assert self.num_fusion_blocks == self.num_reassemble_blocks
#         assert self.num_reassemble_blocks == self.num_post_process_channels
#         self.conv_depth = HeadDepth(self.channels)
        
#         self.fp16_enabled = False
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def depth_pred(self, feat):
#         """Prediction each pixel."""
#         if self.scale_up:
#             output = self.sigmoid(self.conv_depth(feat)) * self.max_depth
#         else:
#             output = self.relu(self.conv_depth(feat)) + self.min_depth
#         return output
#     def forward(self, inputs, img_metas):
#         assert len(inputs) == self.num_reassemble_blocks
#         x = [inp for inp in inputs]
#         x = self.reassemble_blocks(x)
#         x = [self.convs[i](feature) for i, feature in enumerate(x)]
#         out = self.fusion_blocks[0](x[-1])
#         for i in range(1, len(self.fusion_blocks)):
#             out = self.fusion_blocks[i](out, x[-(i + 1)])
#         out = self.project(out)
#         out = self.depth_pred(out)
#         return out
    
class Dinov2ForDepth(torch.nn.Module):
    def __init__(self, backbone_size = "base", image_shape=(224,224), decode_type = 'linear'):
        super().__init__()
        self.backbone_size = backbone_size
        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        self.intermediate_layers = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
            "large": [4, 11, 17, 23],
            "giant": [9, 19, 29, 39],
        }
        
        self.embedding_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        self.image_shape = image_shape
        self.decode_type = decode_type
        self.backbone_arch = self.backbone_archs[self.backbone_size]
        self.n = self.intermediate_layers[self.backbone_size] if decode_type == 'linear4' else 1 #self.intermediate_layers[self.backbone_size][-1]
        self.embedding_dim = self.embedding_dims[self.backbone_size]
        self.backbone_name = f"dinov2_{self.backbone_arch}"
        dinov2 = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.backbone_name)
        for param in dinov2.parameters():
            param.requires_grad = False
        self.dinov2 = dinov2
        
        if self.decode_type == 'linear':
            self.inchannels = [self.embedding_dim]
            self.channels = self.embedding_dim*2
            self.in_index = (0)
            self.input_transform="resize"
        elif self.decode_type == 'linear4':
            self.inchannels = [self.embedding_dim,self.embedding_dim,self.embedding_dim,self.embedding_dim]
            self.channels = self.embedding_dim*8
            self.in_index = (0, 1, 2, 3)
            self.input_transform="resize_concat"
            
        self.decode_head = decode_head_linear(image_shape=self.image_shape, 
                                              input_transform=self.input_transform,
                                              in_index=self.in_index,
                                              in_channels=self.inchannels, 
                                              channels=self.channels)

        self.sig_loss = SigLoss(valid_mask=True, loss_weight=1.0, warm_up=False, loss_name='loss_depth')
        self.gradient_loss = GradientLoss(valid_mask=True, loss_weight=0.5, loss_name='loss_grad')
    
    def save_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        decode_head_tensors = {}
        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.decode_head, torch.nn.DataParallel) or isinstance(self.decode_head, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.decode_head.module.state_dict()
        else:
            state_dict = self.decode_head.state_dict()
        for key, value in state_dict.items():
            decode_head_tensors[key] = value

        merged_dict = {**decode_head_tensors}
        torch.save(merged_dict, filename)

        print('saved parameters to %s.' % filename)
    
    def load_parameters(self, filename: str, device: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename, map_location=device)

        decode_head_dict = self.decode_head.state_dict()
        decode_head_keys = decode_head_dict.keys()

        # load decode head
        decode_head_keys = [k for k in decode_head_keys]
        decode_head_values = [state_dict[k] for k in decode_head_keys]
        decode_head_new_state_dict = {k: v for k, v in zip(decode_head_keys, decode_head_values)}
        decode_head_dict.update(decode_head_new_state_dict)

        self.decode_head.load_state_dict(decode_head_dict)

        print('loaded parameters from %s.' % filename)
        
    def forward(self, pixel_values, depth_gt=None):
        # use frozen features
        feature = self.dinov2.get_intermediate_layers(pixel_values, n=[2, 5, 8, 11], reshape = True, return_class_token = True)
        
        # predict depth with a linear classifier
        pred = self.decode_head(feature)
        # print('pred:', pred.max(), pred.min())
        loss = None
        if depth_gt is not None:
            loss = self.sig_loss(pred.squeeze(), depth_gt.squeeze()) + self.gradient_loss(pred.squeeze(), depth_gt.squeeze())

        return DepthEstimatorOutput(
        loss=loss,
        predicted_depth=pred,
        )
    
class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        # qkv[:, :, :, : self.dim] += new_q
        # qkv[:, :, :, -self.dim:] += new_v
        
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv
    
class EndoDino(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, backbone_size = "base", r=4, image_shape=(224,224), decode_type = 'linear', lora_layer=None):
        super(EndoDino, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        self.backbone_size = backbone_size
        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        self.intermediate_layers = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
            "large": [4, 11, 17, 23],
            "giant": [9, 19, 29, 39],
        }
        
        self.embedding_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        self.backbone_arch = self.backbone_archs[self.backbone_size]
        self.n = self.intermediate_layers[self.backbone_size] if decode_type == 'linear4' else 1 #self.intermediate_layers[self.backbone_size][-1]
        self.embedding_dim = self.embedding_dims[self.backbone_size]
        
        self.backbone_name = f"dinov2_{self.backbone_arch}"
        dinov2 = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.backbone_name)
        self.image_shape = image_shape
        
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(dinov2.blocks)))  # Only apply lora to the image encoder by default
        self.decode_type = decode_type
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # sam_model.image_encoder = dinov2
        # lets freeze first
        for param in dinov2.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(dinov2.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.dinov2 = dinov2
        # The decode depth estimation head
        
        if self.decode_type == 'linear':
            self.inchannels = [self.embedding_dim]
            self.channels = self.embedding_dim*2
            self.in_index = (0)
            self.input_transform="resize"
        elif self.decode_type == 'linear4':
            self.inchannels = [self.embedding_dim,self.embedding_dim,self.embedding_dim,self.embedding_dim]
            self.channels = self.embedding_dim*8
            self.in_index = (0, 1, 2, 3)
            self.input_transform="resize_concat"
            
        self.decode_head = decode_head_linear(image_shape=self.image_shape, 
                                              input_transform=self.input_transform,
                                              in_index=self.in_index,
                                              in_channels=self.inchannels, 
                                              channels=self.channels)
        
        self.sig_loss = SigLoss(valid_mask=True, loss_weight=1.0, warm_up=False, loss_name='loss_depth')
        self.gradient_loss = GradientLoss(valid_mask=True, loss_weight=0.5, loss_name='loss_grad')
    def save_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        decode_head_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.decode_head, torch.nn.DataParallel) or isinstance(self.decode_head, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.decode_head.module.state_dict()
        else:
            state_dict = self.decode_head.state_dict()
        for key, value in state_dict.items():
            decode_head_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **decode_head_tensors}
        torch.save(merged_dict, filename)

        print('saved lora parameters to %s.' % filename)

    def load_parameters(self, filename: str, device: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename, map_location=device)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        decode_head_dict = self.decode_head.state_dict()
        decode_head_keys = decode_head_dict.keys()

        # load decode head
        decode_head_keys = [k for k in decode_head_keys]
        decode_head_values = [state_dict[k] for k in decode_head_keys]
        decode_head_new_state_dict = {k: v for k, v in zip(decode_head_keys, decode_head_values)}
        decode_head_dict.update(decode_head_new_state_dict)

        self.decode_head.load_state_dict(decode_head_dict)

        print('loaded lora parameters from %s.' % filename)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, pixel_values, depth_gt=None):
        feature = self.dinov2.get_intermediate_layers(pixel_values, n=self.n, reshape = True, return_class_token = True, norm=False)
        # out = self.dinov2(pixel_values, is_training=True)
        # print("patch embdding shape:",feature[0][0].shape)
        # print("cls tokens shape:",feature[0][0].shape)
        pred = self.decode_head(feature)
        # print(pred.shape)
        loss = None
        if depth_gt is not None:
            loss = self.sig_loss(pred.squeeze(), depth_gt.squeeze()) + self.gradient_loss(pred.squeeze(), depth_gt.squeeze())
            loss = torch.mean(loss)
        return DepthEstimatorOutput(
        loss=loss,
        predicted_depth=pred,
        )
        
if __name__ == "__main__":
    from endo.scared_dataset import SCAREDDataset
    import torch.utils.data
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    train_dataset = SCAREDDataset(
            "train",
            '/mnt/data-hdd2/Beilei/Dataset/SCARED',
            target_W = 224,
            target_H = 224,
            cap_depth = 150
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=8, shuffle=True, pin_memory=True)
    endodino = EndoDino(backbone_size = "large", r=4, lora_layer=None, image_shape=(224,224), decode_type = 'linear4').to(device = device)
    dinov2 = Dinov2ForDepth(backbone_size = "small", image_shape=(224,224), decode_type = 'linear4').to(device = device)
    
    endodino.train()
    for idx, batch in enumerate(tqdm(train_dataloader)):
        pixel_values = batch["image"].to(device)
        depth_gt = batch["depth"].to(device)
    
        outputs = endodino(pixel_values=pixel_values, depth_gt=depth_gt)
    # outputs = dinov2(pixel_values=pixel_values, depth_gt=depth_gt)
    
    print(outputs.predicted_depth.shape)
    print(outputs.loss)
    
    # for name, param in endodino.named_parameters():
    #     print(name)

    save_path = './endodino.pth'
    endodino.save_parameters(save_path)

    endodino.load_parameters(save_path)
    pass

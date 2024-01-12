import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from transformers.modeling_outputs import DepthEstimatorOutput

def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class GradientLoss(nn.Module):
    """GradientLoss.

    Adapted from https://www.cs.cornell.edu/projects/megadepth/

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
    """

    def __init__(self, valid_mask=True, loss_weight=1.0, max_depth=None, loss_name="loss_grad"):
        super(GradientLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.loss_name = loss_name

        self.eps = 0.001  # avoid grad explode

    def gradientloss(self, input, target):
        input_downscaled = [input] + [input[:: 2 * i, :: 2 * i] for i in range(1, 4)]
        target_downscaled = [target] + [target[:: 2 * i, :: 2 * i] for i in range(1, 4)]

        gradient_loss = 0
        for input, target in zip(input_downscaled, target_downscaled):
            if self.valid_mask:
                mask = target > 0
                if self.max_depth is not None:
                    mask = torch.logical_and(target > 0, target <= self.max_depth)
                N = torch.sum(mask)
            else:
                mask = torch.ones_like(target)
                N = input.numel()
            input_log = torch.log(input + self.eps)
            target_log = torch.log(target + self.eps)
            log_d_diff = input_log - target_log

            log_d_diff = torch.mul(log_d_diff, mask)

            v_gradient = torch.abs(log_d_diff[0:-2, :] - log_d_diff[2:, :])
            v_mask = torch.mul(mask[0:-2, :], mask[2:, :])
            v_gradient = torch.mul(v_gradient, v_mask)

            h_gradient = torch.abs(log_d_diff[:, 0:-2] - log_d_diff[:, 2:])
            h_mask = torch.mul(mask[:, 0:-2], mask[:, 2:])
            h_gradient = torch.mul(h_gradient, h_mask)

            gradient_loss += (torch.sum(h_gradient) + torch.sum(v_gradient)) / N

        return gradient_loss

    def forward(self, depth_pred, depth_gt):
        """Forward function."""

        gradient_loss = self.loss_weight * self.gradientloss(depth_pred, depth_gt)
        return gradient_loss
    
class SigLoss(nn.Module):
    """SigLoss.

        This follows `AdaBins <https://arxiv.org/abs/2011.14141>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(
        self, valid_mask=True, loss_weight=1.0, max_depth=None, warm_up=False, warm_iter=100, loss_name="sigloss"
    ):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.loss_name = loss_name

        self.eps = 0.001  # avoid grad explode

        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]

        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt):
        """Forward function."""

        loss_depth = self.loss_weight * self.sigloss(depth_pred, depth_gt)
        return loss_depth

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
    

class Dinov2ForDepth(torch.nn.Module):
    def __init__(self, backbone_size = "base", image_shape=(224,224), decode_type = 'linear4'):
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
    """In Dinov2 it is implemented as
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
        
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv
    
class SurgicalDINO(nn.Module):
    """Applies low-rank adaptation to Dinov2 model's image encoder.

    Args:
        backbone_size: the pretrained size of dinov2 model
        r: rank of LoRA
        image_shape: input image shape
        decode_type: the decode type of decode head, "linear" or ""

    """

    def __init__(self, backbone_size = "base", r=4, image_shape=(224,224), decode_type = 'linear4', lora_layer=None):
        super(SurgicalDINO, self).__init__()

        assert r > 0
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
        self.n = self.intermediate_layers[self.backbone_size] if decode_type == 'linear4' else 1 
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
        # freeze first
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
        pred = self.decode_head(feature)

        loss = None
        if depth_gt is not None:
            loss = self.sig_loss(pred.squeeze(), depth_gt.squeeze()) + self.gradient_loss(pred.squeeze(), depth_gt.squeeze())
            loss = torch.mean(loss)
        return DepthEstimatorOutput(
        loss=loss,
        predicted_depth=pred,
        )
        
if __name__ == "__main__":
    
    # an example of how to use Surgical-DINO
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # instantiate Surgical-DINO
    surgicaldino = SurgicalDINO(backbone_size="base", r=4, lora_layer=None, image_shape=(224,224), decode_type = 'linear4').to(device = device)
    
    surgicaldino.train()

    # test input image and input gt_depth(optional)
    test = torch.randn([1, 3, 224, 224]).to(device)
    depth_gt = torch.randint(low=10, high=100, size=[1, 1, 224, 224]).to(device)
    
    output = surgicaldino(pixel_values = test, depth_gt = depth_gt)
    print("output max: ", output.predicted_depth.max(), "output min: ", output.predicted_depth.min())
    print("output shape: ", output.predicted_depth.shape)
    print("loss: ", output.loss)
    pass

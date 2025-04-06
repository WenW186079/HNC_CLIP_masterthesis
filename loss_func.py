import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image



class CLIPLoss(nn.Module):
    def __init__(
        self, 
        hard_neg_weight=0.5, 
        lambda_reg=0.01, 
        dynamic_weight=True, 
        min_weight=0, 
        max_weight=10, 
        update_interval=1000,
        num_updates=10,
        compute_stats=False
        ):

        super(CLIPLoss, self).__init__()
        self.hard_neg_weight = hard_neg_weight
        self.lambda_reg = lambda_reg
        self.dynamic_weight = dynamic_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.update_interval = update_interval
        self.num_updates = num_updates
        self.current_step = 0 
        self.compute_stats = compute_stats
    
    def update_step(self, step):
        """Update the current step for dynamic weight calculation."""
        self.current_step = step

    def forward(self, image_features, text_features, logit_scale, original_state_dict=None, model=None):
        if self.dynamic_weight:
            max_update_step = self.update_interval * self.num_updates
            if self.current_step < max_update_step:
                increment = (self.max_weight - self.min_weight) / self.num_updates
                intervals_passed = self.current_step // self.update_interval
                weight = self.min_weight + intervals_passed * increment
            else:
                weight = self.max_weight
        else:
            weight = self.hard_neg_weight
        
        device = image_features.device

        B = image_features.shape[0]

        # Normalize image and text features
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        logits_raw = image_features @ text_features.t()
        logits = logit_scale * logits_raw  # [B, 2B]

        # Create a weight vector for hard negatives: first B are positives (weight 1), next B are scaled.
        weights = torch.cat([
            torch.ones(B, device=device), 
            torch.full((B,), weight, device=device)
        ]).unsqueeze(0)  # Shape: [1, 2B]
        weighted_logits = logits * weights

        # Create target labels: for each image i, the matching positive caption is assumed to be at index i.
        labels = torch.arange(B, device=device)

        # Compute the image-to-text loss using the weighted logits.
        loss_i2t = F.cross_entropy(weighted_logits, labels)
        # Compute the text-to-image loss only on the positive captions (first B rows).
        logits_t2i = logits_raw[:, :B].t()  # shape: [B, B]
        loss_t2i = F.cross_entropy((logit_scale * logits_t2i), labels)
        contrastive_loss = (loss_i2t + loss_t2i) / 2.0
        
        if self.compute_stats:
            positive_score = logits_raw[:, :B].diag() 
            hard_negative_scores = logits_raw[:, B:].diag() 
        else:
            positive_score = torch.tensor(0.0, device=contrastive_loss.device)
            hard_negative_scores = torch.tensor(0.0, device=contrastive_loss.device)

        # Compute regularization loss if original state is provided.
        if original_state_dict is not None and model is not None:
            reg_loss = sum(
                ((param - original_state_dict[name]) ** 2).sum()
                for name, param in model.named_parameters() if param.requires_grad
            )
            total_loss = contrastive_loss + self.lambda_reg * reg_loss
        else:
            reg_loss = torch.tensor(0.0, device=device)
            total_loss = contrastive_loss

        return total_loss, loss_i2t, loss_t2i, reg_loss, positive_score, hard_negative_scores, logit_scale


class StandardCLIPLoss(nn.Module):
    def __init__(self):
        super(StandardCLIPLoss, self).__init__()

    def forward(self, image_features, text_features, logit_scale, original_state_dict=None, model=None):
        B = image_features.shape[0]

        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        similarity_matrix = image_features @ text_features.t()
        positive_score = similarity_matrix.diag()

        logits_per_image = logit_scale * similarity_matrix
        logits_per_text = logits_per_image.t()

        labels = torch.arange(B, device=image_features.device)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        total_loss = (loss_i2t + loss_t2i) / 2.0

        return total_loss, loss_i2t, loss_t2i, positive_score, logit_scale

class CombinedCLIPDPOLoss(nn.Module):
    def __init__(self, beta: float = 1.0, lambda_reg: float = 1e-4, alpha: float = 0.5):
        """
        Combined loss: a weighted sum of CLIP contrastive loss and DPO loss for (image, pos, neg) pairs.
        
        Args:
            beta (float): Scaling factor for the logistic (DPO) loss.
            lambda_reg (float): Weight for the L2 regularization term in DPO loss.
            alpha (float): Weight to balance contrastive loss vs DPO loss.
                           alpha=1.0 means only contrastive loss, alpha=0.0 means only DPO loss.
        """
        super(CombinedCLIPDPOLoss, self).__init__()
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.alpha = alpha

    def forward(self, image_features, text_features, logit_scale, original_state_dict=None, model=None):
        """
        Compute the combined loss for (image, pos, neg) data pairs.
        
        Args:
            image_features (torch.Tensor): [B, D] image embeddings.
            text_features (torch.Tensor): [2B, D] text embeddings where:
                                          - text_features[:B] are positive captions,
                                          - text_features[B:2B] are negative captions.
            logit_scale (torch.Tensor or float): Scaling factor for the dot product similarities.
            original_state_dict (dict, optional): Reference model state dict for L2 regularization.
            model (nn.Module, optional): Model used for L2 regularization.
        
        Returns:
            total_loss, contrastive_loss, dpo_loss, reg_loss (tuple of torch.Tensors):
                - total_loss: The weighted sum of contrastive and DPO losses.
                - contrastive_loss: Standard contrastive loss computed from (image, positive).
                - dpo_loss: The DPO loss component.
                - reg_loss: The L2 regularization term (if applied).
        """
        B = image_features.shape[0]
        # -------------------------------
        # Contrastive Loss (CLIP-style)
        # -------------------------------
        # Use positive captions only for contrastive loss.
        pos_text_features = text_features[:B]  # shape [B, D]
        # Compute similarity matrix between image and positive text features.
        logits_i2t = logit_scale * image_features @ pos_text_features.T  # shape: [B, B]
        logits_t2i = logits_i2t.T  # shape: [B, B]
        
        # Target: each image matches the caption at the same index.
        targets = torch.arange(B, device=image_features.device)
        loss_i2t = F.cross_entropy(logits_i2t, targets)
        loss_t2i = F.cross_entropy(logits_t2i, targets)
        contrastive_loss = (loss_i2t + loss_t2i) / 2.0
        
        # -------------------------------
        # DPO Loss
        # -------------------------------
        # For DPO loss, we assume the text_features contains both positive and negative captions.
        neg_text_features = text_features[B:2*B]  # shape [B, D]
        
        # Compute current similarity scores (dot product) for positive and negative pairs.
        score_current_pos = logit_scale * torch.sum(image_features * pos_text_features, dim=1)  # [B]
        score_current_neg = logit_scale * torch.sum(image_features * neg_text_features, dim=1)  # [B]
        
        # Use the current scores as the reference (detached) to avoid in-place modifications.
        score_ref_pos = score_current_pos.detach().clone()
        score_ref_neg = score_current_neg.detach().clone()
        
        # Compute the delta for each sample.
        # Delta = (current positive - reference positive) - (current negative - reference negative)
        delta = (score_current_pos - score_ref_pos) - (score_current_neg - score_ref_neg)
        
        # Compute the logistic loss.
        dpo_loss = -torch.log(torch.sigmoid(self.beta * delta) + 1e-8)
        dpo_loss = dpo_loss.mean()
        
        # -------------------------------
        # L2 Regularization (for DPO)
        # -------------------------------
        if original_state_dict is not None and model is not None:
            reg_loss = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad and name in original_state_dict:
                    ref_param = original_state_dict[name].to(param.device)
                    reg_loss += ((param - ref_param) ** 2).sum()
        else:
            reg_loss = torch.tensor(0.0, device=image_features.device)
        
        # Add regularization to the DPO loss.
        dpo_loss_total = dpo_loss + self.lambda_reg * reg_loss
        
        # -------------------------------
        # Combine the Losses
        # -------------------------------
        total_loss = self.alpha * contrastive_loss + (1 - self.alpha) * dpo_loss_total
        
        return total_loss, contrastive_loss, dpo_loss_total, reg_loss



from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import distributed as dist

try:
    import torch.distributed.nn
    has_distributed = True
except ImportError:
    has_distributed = False



def _cosine_similarity(x, y, logit_scale=None):
    return x @ y.T if logit_scale is None else logit_scale * x @ y.T


def _cross_entropy_loss(logits, labels, reduction="mean", **kwargs):
    assert reduction in ["mean", "sum", "none"]
    if labels.ndim == 1:
        return F.cross_entropy(logits, labels, reduction=reduction)
    elif labels.ndim == 2:
        # log softmax
        assert reduction == "mean"
        return -1 * torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1).mean()
    else:
        raise ValueError


def _clip_loss(logits_per_image, logits_per_text, labels):
    loss_arg_pairs = [(logits_per_image, labels), (logits_per_text, labels)]
    return sum([_cross_entropy_loss(*arg) for arg in loss_arg_pairs]) / len(loss_arg_pairs)


def gather_features(feature_list, **kwargs):
    gathered_features = [_gather_features(f, **kwargs) for f in feature_list]
    return tuple(gathered_features)


def _gather_features(
    features, local_loss=False, rank=0, world_size=1
):
    if world_size == 1:
        return features

    assert has_distributed, (
        'torch.distributed did not import correctly, please use a PyTorch version with support.'
    )
    gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
    dist.all_gather(gathered_features, features)
    if not local_loss:
        # ensure grads for local rank when all_* features don't have a gradient
        gathered_features[rank] = features
    all_features = torch.cat(gathered_features, dim=0)

    return all_features


class LossFunction(nn.Module):
    """ basic utils for distributed training, implementing all_gather ops """

    def __init__(
        self,
        local_batch_size=0,
        local_loss=False,
        rank=0,
        world_size=1,
        **kwargs
    ):
        super().__init__()
        assert local_batch_size > 0
        self.local_batch_size = local_batch_size
        self.local_loss = local_loss
        self.rank = rank
        self.world_size = world_size
        

    def gather(self, features):
        if not isinstance(features, list):
            features = [features]
        all_features = gather_features(
            features,
            local_loss=self.local_loss,
            rank=self.rank,
            world_size=self.world_size,
        )
        if len(all_features) == 1:
            return all_features[0]
        return all_features


class ClipLoss(LossFunction):
    """ default clip loss; image-text contrastive """

    def __init__(self, cache_labels=False, **kwargs):
        super().__init__(**kwargs)
        # cache state
        self.cache_labels = cache_labels
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = _cosine_similarity(image_features, text_features, logit_scale)

        if text_features.size(0) > image_features.size(0):
            # exclude similarity logits for the negative texts
            logits_per_text = _cosine_similarity(
                text_features[:image_features.size(0)], image_features, logit_scale
            )
        else:
            logits_per_text = _cosine_similarity(text_features, image_features, logit_scale)
        return logits_per_image, logits_per_text

    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        output_dict=False,
        return_logits=False,
        **kwargs,
    ):
        assert return_logits is False
        device = image_features.device
        lbs = self.local_batch_size

        text_features = text_features[:lbs]

        all_img_feats, all_text_feats = self.gather([image_features, text_features])
        logits_per_img, logits_per_txt = self.get_logits(all_img_feats, all_text_feats, logit_scale)
        labels = self.get_ground_truth(device, logits_per_img.shape[0])
        total_loss = _clip_loss(logits_per_img, logits_per_txt, labels)

        return {"clip_loss": total_loss} if output_dict else total_loss


def create_loss(args):
    if args.loss_name == "clip":
        # single contrasive loss, optionally containing hard negative texts
        return ClipLoss(
            local_batch_size=args.batch_size,
            local_loss=False,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
        )

    # # clip loss with only image - positive text pairs plus, separate hard negative loss
    # assert args.loss_name == "fsc-clip"

    # if args.apply_local_neg_loss:
    #     assert args.return_dense_tokens  # enforce models to output token representations
    #     return Local_HNLoss(
    #         # clip loss params
    #         local_batch_size=args.batch_size,
    #         local_loss=args.local_loss,
    #         gather_with_grad=args.gather_with_grad,
    #         cache_labels=True,
    #         rank=args.rank,
    #         world_size=args.world_size,
    #         use_horovod=args.horovod,
    #         # separate neg loss params
    #         apply_global_neg_loss=args.apply_global_neg_loss,
    #         neg_loss_weight=args.neg_loss_weight,  # global neg loss weight
    #         neg_loss_name=args.neg_loss_name,
    #         focal_gamma=args.focal_gamma,
    #         neg_label_smoothing=args.neg_label_smoothing,
    #         # local neg loss params
    #         local_neg_weight=args.local_neg_weight,
    #         sim_normalizer=args.sim_normalizer,
    #         sim_sparsify=args.sim_sparsify
    #     )

    # assert args.apply_global_neg_loss
    # return Global_HNLoss(
    #     # clip loss params
    #     local_batch_size=args.batch_size,
    #     local_loss=args.local_loss,
    #     gather_with_grad=args.gather_with_grad,
    #     cache_labels=True,
    #     rank=args.rank,
    #     world_size=args.world_size,
    #     use_horovod=args.horovod,
    #     # separate neg loss params
    #     apply_global_neg_loss=args.apply_global_neg_loss,
    #     neg_loss_weight=args.neg_loss_weight,  # global neg loss weight
    #     neg_loss_name=args.neg_loss_name,
    #     focal_gamma=args.focal_gamma,
    #     neg_label_smoothing=args.neg_label_smoothing,
    # )

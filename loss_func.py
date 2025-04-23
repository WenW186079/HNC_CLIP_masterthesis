import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class StandardCLIPLoss(nn.Module):
    def __init__(self):
        super(StandardCLIPLoss, self).__init__()

    def _cosine_similarity(self, x, y, logit_scale=None):
        sim = x @ y.t()
        if logit_scale is not None:
            sim = logit_scale * sim
        return sim

    def forward(self, image_features, text_features, logit_scale, original_state_dict=None, model=None):
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        logits_per_image = self._cosine_similarity(image_features, text_features, logit_scale)
        logits_per_text = self._cosine_similarity(text_features, image_features, logit_scale)

        labels = torch.arange(image_features.size(0), device=image_features.device, dtype=torch.long)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        total_loss = (loss_i2t + loss_t2i) / 2.0

        positive_score = (image_features @ text_features.t()).diag()

        return total_loss, loss_i2t, loss_t2i, positive_score, logit_scale

class CLIPLoss(nn.Module):
    def __init__(
        self, 
        hard_neg_weight=0.5, 
        lambda_reg=0.01, 
        dynamic_weight=True, 
        min_weight=0, 
        max_weight=10, 
        update_interval=1000,
        num_updates=10
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

    
    def update_step(self, step):
        """Update the current step for dynamic weight calculation."""
        self.current_step = step
    
    def _cosine_similarity(self, x, y, logit_scale=None):
        sim = x @ y.t()
        if logit_scale is not None:
            sim = logit_scale * sim
        return sim

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

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        pos_text_features = text_features[:B]      # First B: positive captions
        neg_text_features = text_features[B:2*B]     # Next B: negative captions

        pos_logits = self._cosine_similarity(image_features, pos_text_features, logit_scale)  # [B, B]
        neg_logits = self._cosine_similarity(image_features, neg_text_features, logit_scale)  # [B, B]

        pos_weights = torch.ones_like(pos_logits)
    
        # For negative logits, weight only the "own" negative (diagonal) with the computed weight,
        # while leaving off-diagonals unchanged (weight 1).
        neg_weights = torch.ones_like(neg_logits)
        diag_indices = torch.arange(B, device=device)
        neg_weights[diag_indices, diag_indices] = weight

        # Apply the weights element-wise.
        weighted_pos_logits = pos_logits * pos_weights  # Unchanged, since it's all ones.
        weighted_neg_logits = neg_logits * neg_weights

        weighted_logits = torch.cat([weighted_pos_logits, weighted_neg_logits], dim=1) 

        # Create target labels: for each image i, the matching positive caption is assumed to be at index i.
        labels = torch.arange(B, device=device, dtype=torch.long)

        # Compute the image-to-text loss using the weighted logits.
        loss_i2t = F.cross_entropy(weighted_logits, labels)
        loss_t2i = F.cross_entropy(pos_logits.t(), labels)

        contrastive_loss = (loss_i2t + loss_t2i) / 2.0
        
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

        positive_score = pos_logits.diag()        
        hard_negative_scores = neg_logits.diag()
        margin = positive_score.mean() - hard_negative_scores.mean() 

        return total_loss, loss_i2t, loss_t2i, reg_loss, positive_score, hard_negative_scores, logit_scale, margin

class CombinedCLIPDPOLoss(nn.Module):
    def __init__(self, beta: float = 1.0, lambda_reg: float = 1e-4, alpha: float = 0.5):
        super(CombinedCLIPDPOLoss, self).__init__()
        self.beta = beta # scaling factor for the logistic (DPO) loss
        self.lambda_reg = lambda_reg # Weight for the L2 regularization term in DPO loss
        self.alpha = alpha # alpha * contrastive loss + (1-alpha) * DPO loss
        
    def forward(self, image_features, text_features, logit_scale, original_state_dict=None, model=None):
        device = image_features.device
        B = image_features.shape[0]
        
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # Split text features into positive and negative parts
        pos_text_features = text_features[:B]
        neg_text_features = text_features[B:2*B]

        # Contrastive Loss (CLIP-style)
        pos_logits = logit_scale * (image_features @ pos_text_features.t())  # [B, B]
        targets = torch.arange(B, device=device)
        loss_i2t = F.cross_entropy(pos_logits, targets)
        loss_t2i = F.cross_entropy(pos_logits.t(), targets)
        contrastive_loss = (loss_i2t + loss_t2i) / 2.0

        # DPO Loss
        # Compute scores as the diagonal of the dot product similarities
        positive_score = pos_logits.diag()  # True (positive) scores
        neg_logits = logit_scale * (image_features @ neg_text_features.t())
        hard_negative_scores = neg_logits.diag()  # Hard negative scores

        # Compute delta using these scores
        delta = positive_score - hard_negative_scores
        dpo_loss = -torch.log(torch.sigmoid(self.beta * delta) + 1e-8).mean()

        # L2 Regularization
        if original_state_dict is not None and model is not None:
            reg_loss = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad and name in original_state_dict:
                    ref_param = original_state_dict[name].to(param.device)
                    reg_loss += ((param - ref_param) ** 2).sum()
        else:
            reg_loss = torch.tensor(0.0, device=device)
        
        dpo_loss_total = dpo_loss + self.lambda_reg * reg_loss

        # Combine Losses
        total_loss = self.alpha * contrastive_loss + (1 - self.alpha) * dpo_loss_total

        return total_loss, contrastive_loss, dpo_loss_total, reg_loss


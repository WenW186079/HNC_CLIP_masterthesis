import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

'''
Here contains 6 ways of loss:

StandardCLIPLoss - standard clip loss function 
CLIPLossL2 -  using hard negative capture, L2 regularization   
CLIPLossKL - using hard negative capture, KL divergence
DPOCLIPLoss -  using dpo way to deal with (image, pos, neg), with KL divergence
DPOContrastiveCLIPLoss - using dpo way to deal with (image, pos, neg), with KL divergence + standard conrtastive loss
CombinedCLIPDPOLoss - using dpo way to deal with (image, pos, neg), with L2 regularization + standard conrtastive loss

'''
class StandardCLIPLoss(nn.Module):
    def __init__(self):
        super(StandardCLIPLoss, self).__init__()

    def _cosine_similarity(self, x, y, logit_scale=None):
        sim = x @ y.t()
        if logit_scale is not None:
            sim = logit_scale * sim
        return sim

    def forward(self, image_features, text_features, logit_scale, original_state_dict=None, model=None):
        B, device = image_features.size(0), image_features.device
        
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # raw similarity matrix
        raw_sim = self._cosine_similarity(image_features, text_features, logit_scale=None)
        positive_raw = raw_sim.diagonal()                
        mask = ~torch.eye(B, dtype=torch.bool, device=device) 
        negative_raw = raw_sim[mask].view(B, B-1)         

        logits_per_image = self._cosine_similarity(image_features, text_features, logit_scale)
        logits_per_text = self._cosine_similarity(text_features, image_features, logit_scale)

        labels = torch.arange(B, device=device, dtype=torch.long)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        total_loss = (loss_i2t + loss_t2i) / 2.0

        positive_scaled = positive_raw * logit_scale
        negative_scaled = negative_raw * logit_scale

        return total_loss, loss_i2t, loss_t2i, positive_raw, negative_raw, logit_scale, positive_scaled, negative_scaled

class CLIPLossL2(nn.Module):
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

        super(CLIPLossL2, self).__init__()
        self.hard_neg_weight = hard_neg_weight
        self.lambda_reg = lambda_reg
        self.dynamic_weight = dynamic_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.update_interval = update_interval
        self.num_updates = num_updates
        self.current_step = 0 

    def update_step(self, step):
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

        pos_text_features = text_features[:B]      
        neg_text_features = text_features[B:2*B]     

        pos_logits = self._cosine_similarity(image_features, pos_text_features, logit_scale)  # [B, B]
        neg_logits = self._cosine_similarity(image_features, neg_text_features, logit_scale)  # [B, B]

        pos_weights = torch.ones_like(pos_logits)
    
        # For negative logits, weight only the "own" negative (diagonal) with the computed weight,
        # while leaving off-diagonals unchanged (weight 1).
        neg_weights = torch.ones_like(neg_logits)
        diag_indices = torch.arange(B, device=device)
        neg_weights[diag_indices, diag_indices] = weight

        weighted_pos_logits = pos_logits * pos_weights  # all ones
        weighted_neg_logits = neg_logits * neg_weights

        weighted_logits = torch.cat([weighted_pos_logits, weighted_neg_logits], dim=1) 

        # Create target labels: for each image i, the matching positive caption is assumed to be at index i.
        labels = torch.arange(B, device=device, dtype=torch.long)

        
        loss_i2t = F.cross_entropy(weighted_logits, labels)
        loss_t2i = F.cross_entropy(pos_logits.t(), labels)

        contrastive_loss = (loss_i2t + loss_t2i) / 2.0
        
        #  regularization loss 
        if original_state_dict is not None and model is not None:
            reg_loss = sum(
                ((param - original_state_dict[name]) ** 2).sum()
                for name, param in model.named_parameters() if param.requires_grad
            )
            total_loss = contrastive_loss + self.lambda_reg * reg_loss
        else:
            reg_loss = torch.tensor(0.0, device=device)
            total_loss = contrastive_loss

        raw_sim       = self._cosine_similarity(image_features, pos_text_features, logit_scale=None)  
        positive_raw  = raw_sim.diag()                                                       
        random_negative_raw  = raw_sim - raw_sim.diag()    
        hnc_raw       = self._cosine_similarity(image_features, neg_text_features, logit_scale=None).diag() 

        positive_scaled = pos_logits.diag()  
        random_negative_scaled =   pos_logits -  pos_logits.diag()    
        hnc_scaled = neg_logits.diag()
        margin = positive_scaled.mean() - hnc_scaled.mean() 

        return total_loss, contrastive_loss, loss_i2t, loss_t2i, reg_loss, positive_raw, random_negative_raw, hnc_raw, positive_scaled, random_negative_scaled, hnc_scaled, logit_scale, margin, weight
 

class CLIPLossKL(nn.Module):
    def __init__(
        self,
        hard_neg_weight=0.5,
        lambda_reg=0.01,
        temperature=4.0,
        dynamic_weight=False,
        min_weight=0.0,
        max_weight=10.0,
        update_interval=1000,
        num_updates=10
    ):
        super().__init__()
        self.hard_neg_weight = hard_neg_weight
        self.lambda_reg = lambda_reg # weight on the KL distillation term
        self.temperature = temperature # temperature for softening
        self.dynamic_weight = dynamic_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.update_interval = update_interval
        self.num_updates = num_updates
        self.current_step = 0

    def update_step(self, step):
        self.current_step = step

    def _cosine_similarity(self, x, y, logit_scale=None):
        sim = x @ y.t()
        return sim * logit_scale if logit_scale is not None else sim

    def forward(
        self,
        image_features,      
        text_features,       
        logit_scale,         
        teacher_model=None,  
        image_inputs=None,   
        text_inputs=None     
    ):
        if self.dynamic_weight:
            max_steps = self.update_interval * self.num_updates
            if self.current_step < max_steps:
                incr = (self.max_weight - self.min_weight) / self.num_updates
                intervals = self.current_step // self.update_interval
                weight = self.min_weight + intervals * incr
            else:
                weight = self.max_weight
        else:
            weight = self.hard_neg_weight

        device = image_features.device
        B = image_features.size(0)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features  = F.normalize(text_features,  p=2, dim=1)

        pos_text_features = text_features[:B]      
        neg_text_features = text_features[B:2*B]    

        pos_logits = self._cosine_similarity(image_features, pos_text_features, logit_scale)  # [B,B]
        neg_logits = self._cosine_similarity(image_features, neg_text_features, logit_scale)  # [B,B]

        neg_weights = torch.ones_like(neg_logits)
        diag_indices = torch.arange(B, device=device)
        neg_weights[diag_indices, diag_indices] = weight
        weighted_neg_logits = neg_logits * neg_weights
        
        student_logits = torch.cat([pos_logits, weighted_neg_logits], dim=1)   # [B,2B]

        labels = torch.arange(B, device=device, dtype=torch.long)

        # Contrastive losses
        loss_i2t = F.cross_entropy(student_logits, labels)
        loss_t2i = F.cross_entropy(pos_logits.t(), labels)
        contrastive_loss = (loss_i2t + loss_t2i) / 2.0

        # Distillation KL
        if teacher_model is not None:
            teacher_model.eval()
            with torch.no_grad():
                img_t = F.normalize(teacher_model.encode_image(image_inputs), dim=1)
                txt_t = F.normalize(teacher_model.encode_text(text_inputs), dim=1)
                
                pos_t = txt_t[:B]
                neg_t = txt_t[B:2*B]

                # tie the teacher scale for both distributions
                t_scale = teacher_model.logit_scale.detach()
                # compute logits under same scale
                texts_all_s = torch.cat([pos_text_features, neg_text_features], dim=0)
                texts_all_t = torch.cat([pos_t, neg_t], dim=0)
                s_logits_KL = self._cosine_similarity(image_features, texts_all_s) * t_scale
                t_logits_KL = self._cosine_similarity(img_t, texts_all_t) * t_scale

            T = self.temperature
            log_p_s = F.log_softmax(s_logits_KL / T, dim=1)
            p_t     = F.softmax(t_logits_KL / T, dim=1)
            kl_loss = F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)
        else:
            kl_loss = torch.tensor(0.0, device=device)

        total_loss = contrastive_loss + self.lambda_reg * kl_loss

        raw_sim       = self._cosine_similarity(image_features, pos_text_features, logit_scale=None)  
        positive_raw  = raw_sim.diag()                                                       
        random_negative_raw  = raw_sim - raw_sim.diag()    
        hnc_raw       = self._cosine_similarity(image_features, neg_text_features, logit_scale=None).diag() 

        positive_scaled = pos_logits.diag()  
        random_negative_scaled =   pos_logits -  pos_logits.diag()    
        hnc_scaled = neg_logits.diag()
        margin = positive_scaled.mean() - hnc_scaled.mean()
        
        return total_loss, contrastive_loss, loss_i2t, loss_t2i, kl_loss, positive_raw, random_negative_raw, hnc_raw, positive_scaled, random_negative_scaled, hnc_scaled, logit_scale, margin, weight


class DPOContrastiveCLIPLoss(nn.Module):
    def __init__(
        self, 
        beta: float = 0.1,
        alpha: float = 0.5 # Weight for the standard contrastive loss (between 0 and 1)
        ):
        super().__init__()
        self.beta = beta
        self.alpha = alpha

    def forward(
        self,
        image_features,      
        text_features,       
        logit_scale,         
        teacher_model=None,  
        image_inputs=None,   
        text_inputs=None    
    ) -> torch.Tensor:
       
        device = image_features.device
        B = image_features.size(0)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        pos_text_feats = text_features[:B]
        neg_text_feats = text_features[B:2*B]

        pos_logits = self._cosine_similarity(image_features, pos_text_feats, logit_scale)
        neg_logits = self._cosine_similarity(image_features, neg_text_feats, logit_scale)
        
        labels = torch.arange(B, device=device, dtype=torch.long)
        loss_i2t = F.cross_entropy(pos_logits, labels)
        loss_t2i = F.cross_entropy(pos_logits.t(), labels)
        contrastive_loss = (loss_i2t + loss_t2i) / 2.0
        
        pos_logit = pos_logits.diag()
        neg_logit = neg_logits.diag()
    
        teacher_model.eval()
        with torch.no_grad():
            img_ref = F.normalize(teacher_model.encode_image(image_inputs), dim=1)
            txt_ref = F.normalize(teacher_model.encode_text(text_inputs), dim=1)
            logit_scale_ref = teacher_model.logit_scale.detach()

            ref_pos_logits = self._cosine_similarity(img_ref, txt_ref[:B], logit_scale_ref)
            ref_neg_logits = self._cosine_similarity(img_ref, txt_ref[B:2*B], logit_scale_ref)
            
            ref_pos_logit = ref_pos_logits.diag()
            ref_neg_logit = ref_neg_logits.diag()

        delta_pos = pos_logit - ref_pos_logit
        delta_neg = neg_logit - ref_neg_logit

        scores = self.beta * (delta_pos - delta_neg)
        dpo_loss = -torch.log(torch.sigmoid(scores) + 1e-8).mean()

        total_loss = self.alpha * contrastive_loss + (1.0 - self.alpha) * dpo_loss

        positive_scaled = pos_logit
        random_negative_scaled = pos_logits - pos_logit
        hnc_scaled = neg_logit
        margin = positive_scaled.mean() - hnc_scaled.mean()

        return total_loss, contrastive_loss, dpo_loss, positive_scaled, random_negative_scaled, hnc_scaled, logit_scale, margin

    @staticmethod
    def _cosine_similarity(
        image_feats: torch.Tensor,
        text_feats: torch.Tensor,
        logit_scale: torch.Tensor
    ) -> torch.Tensor:
        return logit_scale * image_feats @ text_feats.t()

class DPOCLIPLoss(nn.Module):
    def __init__(
        self, 
        beta: float = 0.1
        ):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        image_features,      
        text_features,       
        logit_scale,         
        teacher_model=None,  
        image_inputs=None,   
        text_inputs=None    
    ) -> torch.Tensor:
       
        device = image_features.device
        B = image_features.size(0)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        pos_text_feats = text_features[:B]
        neg_text_feats = text_features[B:2*B]

        pos_logits = self._cosine_similarity(image_features, pos_text_feats, logit_scale)
        neg_logits = self._cosine_similarity(image_features, neg_text_feats, logit_scale)
        pos_logit = pos_logits.diag()
        neg_logit = neg_logits.diag()
    
        teacher_model.eval()
        with torch.no_grad():
            img_ref = F.normalize(teacher_model.encode_image(image_inputs), dim=1)
            txt_ref = F.normalize(teacher_model.encode_text(text_inputs), dim=1)
            logit_scale_ref = teacher_model.logit_scale.detach()

            ref_pos_logits = self._cosine_similarity(img_ref, txt_ref[:B], logit_scale_ref)
            ref_neg_logits = self._cosine_similarity(img_ref, txt_ref[B:2*B], logit_scale_ref)
            
            ref_pos_logit = ref_pos_logits.diag()
            ref_neg_logit = ref_neg_logits.diag()

        delta_pos = pos_logit - ref_pos_logit
        delta_neg = neg_logit - ref_neg_logit

        scores = self.beta * (delta_pos - delta_neg)
        dpo_loss = -torch.log(torch.sigmoid(scores) + 1e-8).mean()

        positive_scaled = pos_logit
        random_negative_scaled = pos_logits - pos_logit
        hnc_scaled = neg_logit
        margin = positive_scaled.mean() - hnc_scaled.mean()

        return dpo_loss, positive_scaled, random_negative_scaled, hnc_scaled, logit_scale, margin

    @staticmethod
    def _cosine_similarity(
        image_feats: torch.Tensor,
        text_feats: torch.Tensor,
        logit_scale: torch.Tensor
    ) -> torch.Tensor:
        return logit_scale * image_feats @ text_feats.t()

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
        
        pos_text_features = text_features[:B]
        neg_text_features = text_features[B:2*B]

        # Contrastive Loss
        pos_logits = logit_scale * (image_features @ pos_text_features.t())  # [B, B]
        targets = torch.arange(B, device=device)
        loss_i2t = F.cross_entropy(pos_logits, targets)
        loss_t2i = F.cross_entropy(pos_logits.t(), targets)
        contrastive_loss = (loss_i2t + loss_t2i) / 2.0

        # DPO Loss
        positive_score = pos_logits.diag()  
        neg_logits = logit_scale * (image_features @ neg_text_features.t())
        hard_negative_scores = neg_logits.diag()  

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

        total_loss = self.alpha * contrastive_loss + (1 - self.alpha) * dpo_loss_total

        return total_loss, contrastive_loss, dpo_loss, reg_loss

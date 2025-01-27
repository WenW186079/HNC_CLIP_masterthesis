import torch
import torch.nn as nn
import torch.nn.functional as F

def safe_exp(tensor, clamp_min=-10, clamp_max=10):
    """Clamp tensor values before exponentiation to prevent overflow."""
    return torch.exp(tensor.clamp(min=clamp_min, max=clamp_max))

class HNC_Loss(nn.Module):
    def __init__(self, temperature=0.07, hard_negative_weight=1.0, l2_reg_weight=1e-3, ref_model=None):
        """
        Contrastive Loss with Hard Negative and L2 Regularization for CLIP.
        
        Args:
            temperature (float): Temperature parameter (\( \tau \)) for scaling cosine similarity.
            hard_negative_weight (float): Weight (\( w_i \)) for hard negative samples.
            l2_reg_weight (float): Weight (\( \lambda \)) for L2 regularization.
            ref_model (torch.nn.Module): Reference model (e.g., pretrained CLIP) for L2 regularization.
        """
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.base_hard_negative_weight = hard_negative_weight
        self.l2_reg_weight = l2_reg_weight
        self.ref_model = ref_model
        if self.ref_model:
            self.ref_model.eval()  
            self.ref_model_params = {name: param.clone().detach().cpu() for name, param in self.ref_model.named_parameters()}

    def update_hard_negative_weight(self, epoch, total_epochs):
        """
        Dynamically update the hard negative weight based on the current epoch.
        """
        self.hard_negative_weight = self.base_hard_negative_weight * (1 + epoch / total_epochs)


    def forward(self, image_embeddings, pos_text_embeddings, neg_text_embeddings, model):
    
        # print('========== Before Normalize=========')
        # print(f"Shape of image_embeddings: {image_embeddings.shape}")  # [batch_size, embedding_dim]
        # print(f"Shape of pos_text_embeddings: {pos_text_embeddings.shape}")  # [batch_size, embedding_dim]
        # print(f"Shape of neg_text_embeddings: {neg_text_embeddings.shape}")  # [batch_size, embedding_dim]

        # print(f"image_embeddings: {image_embeddings}")
        # print(f"pos_text_embeddings: {pos_text_embeddings}")
        # print(f"neg_text_embeddings: {neg_text_embeddings}")

        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        pos_text_embeddings = F.normalize(pos_text_embeddings, dim=-1)
        neg_text_embeddings = F.normalize(neg_text_embeddings, dim=-1)
        
        # print('========== After Normalize=========')
        # print(f"Shape of image_embeddings: {image_embeddings.shape}")  # [batch_size, embedding_dim]
        # print(f"Shape of pos_text_embeddings: {pos_text_embeddings.shape}")  # [batch_size, embedding_dim]
        # print(f"Shape of neg_text_embeddings: {neg_text_embeddings.shape}")  # [batch_size, embedding_dim]

        # print(f"image_embeddings: {image_embeddings}")
        # print(f"pos_text_embeddings: {pos_text_embeddings}")
        # print(f"neg_text_embeddings: {neg_text_embeddings}")

        # Compute similarity matrices
        sim_image_to_pos = torch.mm(image_embeddings, pos_text_embeddings.t()) / self.temperature  # [batch_size, batch_size]
        sim_image_to_neg = torch.mm(image_embeddings, neg_text_embeddings.t()) / self.temperature  # [batch_size, batch_size]
    
        sim_text_to_image_pos = sim_image_to_pos.t()  
        sim_text_to_image_neg = sim_image_to_neg.t()

        # Extract diagonals
        diag_image_pos = torch.diag(sim_image_to_pos)  # Positive similarities (image-to-text)
        diag_image_neg = torch.diag(sim_image_to_neg)  # Hard negative similarities (image-to-text)
        diag_text_pos = torch.diag(sim_text_to_image_pos)  # Positive similarities (text-to-image)
        diag_text_neg = torch.diag(sim_text_to_image_neg)  # Hard negative similarities (text-to-image)

        # Compute denominators for image-to-text loss
        denom_image_to_text = (
            safe_exp(diag_image_pos) +
            torch.sum(safe_exp(sim_image_to_pos), dim=1) - safe_exp(diag_image_pos) +  
            self.hard_negative_weight * safe_exp(diag_image_neg) +
            torch.sum(safe_exp(sim_image_to_neg), dim=1) - safe_exp(diag_image_neg)  
        )

        # Compute denominators for text-to-image loss
        denom_text_to_image = (
            safe_exp(diag_text_pos) +
            torch.sum(safe_exp(sim_text_to_image_pos), dim=1) - safe_exp(diag_text_pos) +  
            self.hard_negative_weight * safe_exp(diag_text_neg) +
            torch.sum(safe_exp(sim_text_to_image_neg), dim=1) - safe_exp(diag_text_neg) 
        )

        denom_image_to_text = denom_image_to_text + 1e-8
        denom_text_to_image = denom_text_to_image + 1e-8

        # Compute losses
        loss_image_to_text = -torch.mean(torch.log(safe_exp(diag_image_pos) / denom_image_to_text))
        loss_text_to_image = -torch.mean(torch.log(safe_exp(diag_text_pos) / denom_text_to_image))
        contrastive_loss = 0.5 * (loss_image_to_text + loss_text_to_image)

        # L2 regularization
        l2_loss = 0.0
        if self.ref_model is not None:
            for name, current_param in model.named_parameters():
                if name in self.ref_model_params:
                    ref_param = self.ref_model_params[name].to(current_param.device)  
                    l2_loss += torch.sum((current_param - ref_param) ** 2)

        total_loss = contrastive_loss + self.l2_reg_weight * l2_loss

        # print(f"diag_image_pos: {diag_image_pos}")
        # print(f"diag_image_neg: {diag_image_neg}")
        # print(f"denom_image_to_text: {denom_image_to_text}")
        # print(f"denom_text_to_image: {denom_text_to_image}")
        # print(f"loss_image_to_text: {loss_image_to_text}, loss_text_to_image: {loss_text_to_image}")
        # print(f"total_loss: {total_loss}")

        return total_loss


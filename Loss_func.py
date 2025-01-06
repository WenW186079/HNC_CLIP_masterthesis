import torch
from torch import nn
from torch.nn.functional import cosine_similarity


class HNC_Loss(nn.Module):
    def __init__(self, clip_params, alpha=0.5, tau=0.07, lambda_=0.01):
        super(HNC_Loss, self).__init__()
        self.alpha = alpha  # Controls HNC negative sample impact
        self.tau = tau  # Temperature parameter
        self.lambda_ = lambda_  # L2 regularization
        self.clip_params = clip_params  # Original CLIP parameters (for regularization)
        
    def forward(self, v_i, u_i_pos, u_i_hnc_neg, u_i_batch_neg, params):
        """
        Compute loss using in-batch negatives and weighted HNC negatives.
        """
        s_ii = cosine_similarity(v_i, u_i_pos) / self.tau  # Positive pair
        s_i_hnc = cosine_similarity(v_i, u_i_hnc_neg) / self.tau  # HNC negative
        s_i_batch_neg = cosine_similarity(v_i.unsqueeze(1), u_i_batch_neg).mean(dim=1) / self.tau  # In-batch negatives

        # Image-to-text loss component
        loss_img_to_text = -torch.log(
            torch.exp(s_ii) / (torch.exp(s_ii) + self.alpha * torch.exp(s_i_hnc) + torch.exp(s_i_batch_neg).sum(dim=0), min=1e-10)
        ).mean()

        # Text-to-image loss component
        loss_text_to_img = -torch.log(
            torch.clamp(torch.exp(s_ii) / torch.exp(s_ii).sum(dim=0), min=1e-10)
        ).mean()

        # L2 regularization term
        l2_term = 0
        for name, param in params.items():
            if param.requires_grad and "visual" in name:  
                original_param = self.clip_params[name]
                l2_term += ((param - original_param) ** 2).sum()

        l2_term *= self.lambda_

        total_loss = loss_img_to_text + loss_text_to_img + l2_term
        return total_loss

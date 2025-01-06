import torch
from torch import nn
from torch.nn.functional import cosine_similarity


class HNC_Loss(nn.Module):
    def __init__(self, fisher_matrix, clip_params, alpha=0.5, tau=0.07, lambda_=0.1):
        super(HNC_Loss, self).__init__()
        self.alpha = alpha  # Controls HNC negative sample impact
        self.tau = tau  # Temperature parameter
        self.lambda_ = lambda_  # EWC regularization
        self.fisher_matrix = fisher_matrix  # Precomputed Fisher Information
        self.clip_params = clip_params  # Original CLIP parameters (for regularization)
        
    def forward(self, v_i, u_i_pos, u_i_neg, params, source):
        s_ii = cosine_similarity(v_i, u_i_pos) / self.tau  # Positive pair
        s_i_neg = cosine_similarity(v_i.unsqueeze(1), u_i_neg).mean(dim=1) / self.tau  # Negative pairs

        # Convert source into weight tensors (alpha for "hnc", 1.0 for "random")
        source_weights = torch.tensor([self.alpha if s == "hnc" else 1.0 for s in source], device=v_i.device)

        # Apply weights to negative pairs
        weighted_neg = torch.exp(s_i_neg) * source_weights

        # Image-to-text loss component
        loss_img_to_text = -torch.log(
            torch.exp(s_ii) / (torch.exp(s_ii) + weighted_neg.sum(dim=0))
        ).mean()

        # Text-to-image loss component
        loss_text_to_img = -torch.log(
            torch.exp(s_ii) / torch.exp(s_ii).sum(dim=0)
        ).mean()

        # EWC regularization term
        ewc_term = 0
        for name, param in params.items():
            if name in self.fisher_matrix:
                original_param = self.clip_params[name]
                fisher_info = self.fisher_matrix[name]
                ewc_term += fisher_info * ((param - original_param) ** 2).sum()

        ewc_term *= self.lambda_

        total_loss = loss_img_to_text + loss_text_to_img + ewc_term
        return total_loss

import torch
from torch import nn
from torch.nn.functional import cosine_similarity


class HNC_Loss(nn.Module):
    def __init__(self, fisher_matrix, clip_params, alpha=0.5, tau=0.07, lambda_=0.1):
        super(HNC_Loss, self).__init__()
        self.alpha = alpha
        self.tau = tau
        self.lambda_ = lambda_
        self.fisher_matrix = fisher_matrix  
        self.clip_params = clip_params 
        
    def forward(self, v_i, u_i_pos, u_i_hnc_neg, u_i_rand_neg, params):
        # Image-to-text loss component
        s_ii = cosine_similarity(v_i, u_i_pos) / self.tau
        s_i_hnc = cosine_similarity(v_i, u_i_hnc_neg) / self.tau
        s_ij_neg = cosine_similarity(v_i.unsqueeze(1), u_i_rand_neg).mean(dim=1) / self.tau

        loss_img_to_text = -torch.log(
            torch.exp(s_ii) / (torch.exp(s_ii) + self.alpha * torch.exp(s_i_hnc) + torch.exp(s_ij_neg))
        ).mean()

        # Text-to-image loss component
        s_ki = cosine_similarity(v_i, u_i_pos) / self.tau
        loss_text_to_img = -torch.log(
            torch.exp(s_ii) / torch.exp(s_ki).sum(dim=0)
        ).mean()

        # EWC regularization
        ewc_term = 0
        for name, param in params.items():
            if name in self.fisher_matrix:
                original_param = self.clip_params[name]
                fisher_info = self.fisher_matrix[name]
                ewc_term += fisher_info * ((param - original_param) ** 2).sum()

        ewc_term *= self.lambda_

        total_loss = loss_img_to_text + loss_text_to_img + ewc_term
        return total_loss

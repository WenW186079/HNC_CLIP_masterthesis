import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import clip
import json

from load_data import HNCCLIPDataset

class ContrastiveHNCWithL2Loss(nn.Module):
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
        self.l2_reg_weight = l2_reg_weight
        self.ref_model = ref_model
        if self.ref_model:
            self.ref_model.eval()  

    def forward(self, paired_batch, model):
        """
        Compute contrastive loss using paired batch data.

        Args:
            paired_batch (list): List of (image, positive_caption, hard_negative_caption) tuples.
            model (torch.nn.Module): The model being trained.

        Returns:
            loss (Tensor): Total loss value.
        """
        # Extract image and text embeddings from paired_batch
        image_embeddings = torch.stack([pair[0] for pair in paired_batch])  # [batch_size, embedding_dim]
        pos_text_embeddings = torch.stack([pair[1] for pair in paired_batch])  # [batch_size, embedding_dim]
        neg_text_embeddings = torch.stack([pair[2] for pair in paired_batch])  # [batch_size, embedding_dim]

        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        pos_text_embeddings = F.normalize(pos_text_embeddings, dim=-1)
        neg_text_embeddings = F.normalize(neg_text_embeddings, dim=-1)

        # Compute similarity matrices
        sim_image_to_pos = torch.mm(image_embeddings, pos_text_embeddings.t()) / self.temperature  # [batch_size, batch_size]
        sim_image_to_neg = torch.mm(image_embeddings, neg_text_embeddings.t()) / self.temperature  # [batch_size, batch_size]
        sim_text_to_image_pos = sim_image_to_pos.t()  # Transpose for text-to-image similarities
        sim_text_to_image_neg = sim_image_to_neg.t()

        # Extract diagonals
        diag_image_pos = torch.diag(sim_image_to_pos)  # Positive similarities (image-to-text)
        diag_image_neg = torch.diag(sim_image_to_neg)  # Hard negative similarities (image-to-text)
        diag_text_pos = torch.diag(sim_text_to_image_pos)  # Positive similarities (text-to-image)
        diag_text_neg = torch.diag(sim_text_to_image_neg)  # Hard negative similarities (text-to-image)

        # Compute denominators for image-to-text loss
        denom_image_to_text = (
            torch.exp(diag_image_pos) +
            torch.sum(torch.exp(sim_image_to_pos), dim=1) - torch.exp(diag_image_pos) +  # Random negatives
            self.hard_negative_weight * torch.exp(diag_image_neg) +
            torch.sum(torch.exp(sim_image_to_neg), dim=1) - torch.exp(diag_image_neg)  # Hard negatives
        )

        # Compute denominators for text-to-image loss
        denom_text_to_image = (
            torch.exp(diag_text_pos) +
            torch.sum(torch.exp(sim_text_to_image_pos), dim=1) - torch.exp(diag_text_pos) +  # Random negatives
            self.hard_negative_weight * torch.exp(diag_text_neg) +
            torch.sum(torch.exp(sim_text_to_image_neg), dim=1) - torch.exp(diag_text_neg)  # Hard negatives
        )

        # Compute losses
        loss_image_to_text = -torch.mean(torch.log(torch.exp(diag_image_pos) / denom_image_to_text))
        loss_text_to_image = -torch.mean(torch.log(torch.exp(diag_text_pos) / denom_text_to_image))

        # Combine contrastive losses
        contrastive_loss = 0.5 * (loss_image_to_text + loss_text_to_image)

        # L2 regularization
        l2_loss = 0.0
        if self.ref_model is not None:
            for name, current_param in model.named_parameters():
                if name in dict(self.ref_model.named_parameters()):
                    ref_param = dict(self.ref_model.named_parameters())[name]
                    l2_loss += torch.sum((current_param - ref_param) ** 2)

        total_loss = contrastive_loss + self.l2_reg_weight * l2_loss
        return total_loss


train_json_file_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/HNC/hnc_train_sampled_1_percent.json'
val_json_file_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/HNC/hnc_val_sampled_1_percent.json'
image_folder_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/gqa_dataset/images/images'

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
ref_model = clip_model.eval()  # Reference model for L2 regularization
tokenizer = clip.tokenize

# Load annotations
with open(train_json_file_path, 'r') as f:
    train_annotations = json.load(f)

# Initialize dataset and DataLoader
dataset = HNCCLIPDataset(
    annotations=train_annotations,
    image_folder=image_folder_path,
    transform=preprocess
)
data_loader = DataLoader(dataset, batch_size=3, shuffle=True)

# Initialize the loss function
loss_fn = ContrastiveHNCWithL2Loss(
    temperature=0.07,
    hard_negative_weight=1.0,
    l2_reg_weight=1e-3,
    ref_model=ref_model
)

# Optimizer for vision encoder only
vision_params = [param for name, param in clip_model.named_parameters() if "visual" in name and param.requires_grad]
optimizer = torch.optim.Adam(vision_params, lr=1e-4)

# Training loop
for epoch in range(5):  # Number of epochs
    clip_model.train()
    for batch in data_loader:
        # Generate paired data using the unique pair function
        paired_batch = HNCCLIPDataset.pair_data_tensor_unique(batch, tokenizer)

        # Compute embeddings for images and captions
        paired_batch = [
            (
                clip_model.encode_image(image.unsqueeze(0)).squeeze(0),
                clip_model.encode_text(pos_caption).squeeze(0),
                clip_model.encode_text(neg_caption).squeeze(0),
            )
            for image, pos_caption, neg_caption in paired_batch
        ]

        # Compute the loss
        loss = loss_fn(paired_batch, clip_model)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

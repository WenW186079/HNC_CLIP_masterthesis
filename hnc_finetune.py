import logging
import wandb
import torch

def preprocess_text_and_images(batch, processor, device):
    """
    Preprocess the batch data: images, positive captions, and negative captions.
    """
    image_paths, pos_captions, neg_captions = batch

    # Load and preprocess images
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

    # Use CLIPProcessor for preprocessing
    inputs = processor(
        images=images,
        text=pos_captions + neg_captions,
        return_tensors="pt",
        padding=True
    ).to(device)

    pos_text_inputs = inputs["input_ids"][:len(pos_captions)]
    neg_text_inputs = inputs["input_ids"][len(pos_captions):]

    return inputs["pixel_values"], pos_text_inputs, neg_text_inputs


def train_clip_model(model, processor, data_loader, loss_fn, optimizer, num_epochs, device):
    """
    Train the CLIP model's vision encoder.

    Args:
        model (CLIPModel): Pretrained CLIP model from Hugging Face.
        processor (CLIPProcessor): Preprocessor for CLIP model.
        data_loader (DataLoader): DataLoader for training data.
        loss_fn (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer for the vision encoder.
        num_epochs (int): Number of epochs to train for.
        device (str): Device to train on (e.g., "cuda" or "cpu").

    Returns:
        None
    """
    
    wandb.init(
        project="fine-tune-hnc-clip", 
        name="clip-vision-fine-tuning",  
        config={
            "num_epochs": num_epochs,
            "learning_rate": optimizer.defaults.get("lr", "unknown"),
            "batch_size": data_loader.batch_size,
        },
    )

    model.train()

    # Freeze text encoder and projection layers
    for param in model.text_model.parameters():
        param.requires_grad = False
    for param in model.visual_projection.parameters():
        param.requires_grad = False

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs} starts.")
        total_loss = 0.0

        for batch_idx, batch in enumerate(data_loader):
            # Preprocess batch data
            pixel_values, pos_text_inputs, neg_text_inputs = preprocess_text_and_images(batch, processor, device)

            # Forward pass for image and text embeddings
            image_embeddings = model.get_image_features(pixel_values)
            with torch.no_grad():
                pos_text_embeddings = model.get_text_features(pos_text_inputs)  
                neg_text_embeddings = model.get_text_features(neg_text_inputs)

            loss = loss_fn(image_embeddings, pos_text_embeddings, neg_text_embeddings, model)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            wandb.log({"batch_loss": loss.item(), "batch_idx": batch_idx, "epoch": epoch + 1})

            if batch_idx % 10 == 0:
                logging.info(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}.")

        avg_loss = total_loss / len(data_loader)
        logging.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss}.")
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})

    wandb.finish()
    logging.info("Training completed.")

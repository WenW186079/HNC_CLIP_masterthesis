import logging
import wandb
import torch
from PIL import Image  
import math
import time
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub import HfApi, HfFolder, Repository

def preprocess_text_and_images(batch, processor, device):
    """
    Preprocess the batch data: images, positive captions, and negative captions.
    """
    image_paths, pos_captions, neg_captions = batch

    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

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

    """
    num_samples = len(data_loader.dataset)  
    batch_size = data_loader.batch_sampler.batch_size if hasattr(data_loader.batch_sampler, 'batch_size') else data_loader.batch_size
    if batch_size is None:
        raise ValueError("Batch size could not be determined. Ensure your DataLoader or Sampler specifies it.")

    num_batches = math.ceil(num_samples / batch_size)  

    wandb.init(
        project="fine-tune-hnc-clip", 
        name="clip-vision-fine-tuning",  
        config={
            "num_epochs": num_epochs,
            "learning_rate": optimizer.defaults.get("lr", "unknown"),
            "batch_size": batch_size,
            "num_batches": num_batches,
            "num_samples": num_samples
        },
    )

    model.train()

    # Freeze text encoder and projection layers
    for param in model.text_model.parameters():
        param.requires_grad = False
    for param in model.visual_projection.parameters():
        param.requires_grad = False

    total_training_time = 0

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs} starts. Total Batches: {num_batches}.")
        total_loss = 0.0
        start_epoch_time = time.time() 

        for batch_idx, batch in enumerate(data_loader):
            batch_start_time = time.time()
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
                logging.info(f"Epoch {epoch + 1}, Batch {batch_idx}/{num_batches}, Loss: {loss.item()}.")

            batch_time = time.time() - batch_start_time
            wandb.log({"batch_time": batch_time})

        avg_loss = total_loss / len(data_loader)
        epoch_time = time.time() - start_epoch_time
        total_training_time += epoch_time

        logging.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss}. Epoch Time: {epoch_time:.2f} seconds. Total Batches: {num_batches}.")
        wandb.log({"epoch_loss": avg_loss, "epoch_time": epoch_time, "epoch": epoch + 1})

    logging.info(f"Training completed. Total training time: {total_training_time:.2f} seconds.")
    wandb.log({"total_training_time": total_training_time})
    wandb.finish()

def push_to_hub(model, processor, repo_name, output_dir, commit_message="Upload fine-tuned CLIP model"):
    """
    Push the fine-tuned model and processor to the Hugging Face Hub.

    """
    # Save model and processor locally
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logging.info(f"Model saved to {output_dir}")

    # Push to Hugging Face Hub
    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{repo_name}"

    logging.info(f"Pushing model to Hugging Face Hub: {repo_id}")
    model.push_to_hub(repo_id, commit_message=commit_message)
    processor.push_to_hub(repo_id, commit_message=commit_message)
    logging.info("Model and processor successfully pushed to the Hugging Face Hub.")

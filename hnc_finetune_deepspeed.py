import logging
import wandb
import torch
from PIL import Image  
import math
import time
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub import HfApi, HfFolder, Repository
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
'''

This version added:
    train with deepspeed
    SGDR: STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS

'''

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


def train_clip_model(model_engine, processor, data_loader, loss_fn, optimizer, num_epochs, device,learning_rate):
    """
    Train the CLIP model's vision encoder.

    """
    num_samples = len(data_loader.dataset)  
    num_batches = len(data_loader) 

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=2, 
        T_mult=2, 
        eta_min=1e-6
    )

    wandb.init(
        project="fine-tune-hnc-clip", 
        name="clip-vision-fine-tuning",  
        config={
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "num_batches": num_batches,
            "num_samples": num_samples
        },
    )

    model_engine.train()

    # Freeze text encoder and projection layers
    for param in model_engine.text_model.parameters():
        param.requires_grad = False
    for param in model_engine.visual_projection.parameters():
        param.requires_grad = False

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs} begins.")
        total_loss = 0.0
    
        for step, batch in enumerate(data_loader):

            # Preprocess batch data
            pixel_values, pos_text_inputs, neg_text_inputs = preprocess_text_and_images(batch, processor, device)

            # Forward pass for image and text embeddings
            image_embeddings = model_engine.get_image_features(pixel_values)
            with torch.no_grad():
                pos_text_embeddings = model_engine.get_text_features(pos_text_inputs)  
                neg_text_embeddings = model_engine.get_text_features(neg_text_inputs)

            loss = loss_fn(image_embeddings, pos_text_embeddings, neg_text_embeddings, model_engine)

            # Backpropagation
            optimizer.zero_grad()
            model_engine.backward(loss)
            model_engine.step()
            scheduler.step(epoch + step / len(data_loader)) 

            total_loss += loss.item()
            wandb.log({"batch_loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]})


            if step % 100 == 0:
                logging.info(f"Epoch {epoch + 1}, Step {step}/{num_samples}, Loss: {loss.item()}")

        avg_loss = total_loss / num_batches
        logging.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss}")
        wandb.log({"epoch_loss": avg_loss})
        
        # Save cache after each epoch
        epoch_cache_dir = f"./epoch_cache/epoch_{epoch + 1}"
        os.makedirs(epoch_cache_dir, exist_ok=True)
        model_engine.save_pretrained(epoch_cache_dir)
        processor.save_pretrained(epoch_cache_dir)
        logging.info(f"Model and processor saved to {epoch_cache_dir} after epoch {epoch + 1}.")

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
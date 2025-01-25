import logging
import wandb
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup
from huggingface_hub import HfApi
import deepspeed.comm as dist
import os

'''

This version added:
    train with deepspeed
    SGDR: STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS

'''

# def preprocess_text_and_images(batch, processor, device):
#     """
#     Preprocess the batch data: images, positive captions, and negative captions.
#     """
#     image_paths, pos_captions, neg_captions = batch
#     images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

#     inputs = processor(
#         images=images,
#         text=pos_captions + neg_captions,
#         return_tensors="pt",
#         padding=True
#     ).to(device)

#     pos_text_inputs = inputs["input_ids"][:len(pos_captions)]
#     neg_text_inputs = inputs["input_ids"][len(pos_captions):]

#     return inputs["pixel_values"], pos_text_inputs, neg_text_inputs

def preprocess_text_and_images(batch, processor, device):
    """
    Preprocess the batch data: images, positive captions, and negative captions.
    Remove duplicate image paths within the batch.
    """
    image_paths, pos_captions, neg_captions = batch
    # logging.info(f"Rank {dist.get_rank()} Original batch size: {len(image_paths)}")
    
    # Remove duplicate image paths and corresponding captions
    seen = set()
    unique_image_paths = []
    unique_pos_captions = []
    unique_neg_captions = []

    for i, image_path in enumerate(image_paths):
        if image_path not in seen:
            seen.add(image_path)
            unique_image_paths.append(image_path)
            unique_pos_captions.append(pos_captions[i])
            unique_neg_captions.append(neg_captions[i])

    # logging.info(f"Rank {dist.get_rank()} Unique batch size: {len(unique_image_paths)}")

    images = [Image.open(image_path).convert("RGB") for image_path in unique_image_paths]
    inputs = processor(
        images=images,
        text=unique_pos_captions + unique_neg_captions,
        return_tensors="pt",
        padding=True
    ).to(device)

    pos_text_inputs = inputs["input_ids"][:len(unique_pos_captions)]
    neg_text_inputs = inputs["input_ids"][len(unique_pos_captions):]

    return inputs["pixel_values"], pos_text_inputs, neg_text_inputs

def train_clip_model(model_engine, processor, data_loader, sampler, loss_fn, optimizer, num_epochs, device,learning_rate):
    """
    Train the CLIP model's vision encoder.

    """
    num_samples = len(data_loader.dataset)  
    batch_size = data_loader.batch_sampler.batch_size if hasattr(data_loader.batch_sampler, 'batch_size') else data_loader.batch_size
    num_batches = len(data_loader) 

    if dist.get_rank() == 0: 
        wandb.init(
            project="fine-tune-hnc-clip",
            name="clip-vision-fine-tuning",
            reinit=True,
            config={
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_batches": num_batches,
                "num_samples": num_samples,
            },
        )

    model_engine.train()

    # Freeze text encoder and projection layers
    for param in model_engine.text_model.parameters():
        param.requires_grad = False
    for param in model_engine.visual_projection.parameters():
        param.requires_grad = False

    best_loss = float('inf')

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)

        if dist.get_rank() == 0:
            logging.info(f"Epoch {epoch + 1}/{num_epochs} begins.")
       
        total_loss = 0.0
    
        for step, batch in enumerate(data_loader):
            # logging.info(f"Rank {dist.get_rank()} is processing batch {step} with size {len(batch[0])}")

            # Preprocess batch data
            pixel_values, pos_text_inputs, neg_text_inputs = preprocess_text_and_images(batch, processor, device)
            # logging.info("Finished preprocessing batch.")
            
            # Forward pass for image and text embeddings
            image_embeddings = model_engine.get_image_features(pixel_values)
            # logging.info("Finished forward pass.")
            with torch.no_grad():
                pos_text_embeddings = model_engine.get_text_features(pos_text_inputs)  
                neg_text_embeddings = model_engine.get_text_features(neg_text_inputs)

            loss = loss_fn(image_embeddings, pos_text_embeddings, neg_text_embeddings, model_engine)
            # logging.info("Computed loss.")

            # Backpropagation
            optimizer.zero_grad()
            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            if dist.get_rank() == 0:
                wandb.log({"batch_loss": loss.item(), "current_lr": optimizer.param_groups[0]["lr"]})

            if dist.get_rank() == 0 and step % 100 == 0:
                logging.info(f"Epoch {epoch + 1}, Step {step}/{num_batches}, Loss: {loss.item():.4f}")


        avg_loss = total_loss / num_batches
        if dist.get_rank() == 0:
            logging.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss}")
            wandb.log({"epoch_loss": avg_loss})
        
        # Save cache after each epoch
        epoch_cache_dir = f"./epoch_cache/epoch_{epoch + 1}"
        os.makedirs(epoch_cache_dir, exist_ok=True)
        model_engine.save_pretrained(epoch_cache_dir)
        processor.save_pretrained(epoch_cache_dir)
        logging.info(f"Model and processor saved to {epoch_cache_dir} after epoch {epoch + 1}.")

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_dir = f"./best_model/"
            os.makedirs(best_model_dir, exist_ok=True)
            model_engine.save_pretrained(best_model_dir)
            processor.save_pretrained(best_model_dir)
            logging.info(f"Best model saved to {best_model_dir}.")

    wandb.finish()

    # Push the best model to Hugging Face
    if dist.get_rank() == 0:

        best_model = CLIPModel.from_pretrained(best_model_dir)
        best_processor = CLIPProcessor.from_pretrained(best_model_dir)

        push_to_hub(
            model=best_model,
            processor=best_processor,
            repo_name='best_model_hnc'
        )


def push_to_hub(model, processor, repo_name):
    """
    Push the fine-tuned model and processor to the Hugging Face Hub.

    """
    # Push to Hugging Face Hub
    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{repo_name}"

    logging.info(f"Pushing model to Hugging Face Hub: {repo_id}")
    model.push_to_hub(repo_id, commit_message="Upload fine-tuned CLIP model")
    processor.push_to_hub(repo_id, commit_message="Upload fine-tuned CLIP model")
    logging.info("Model and processor successfully pushed to the Hugging Face Hub.")

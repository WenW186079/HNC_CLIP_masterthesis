import logging
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from PIL import Image
from huggingface_hub import HfApi
import deepspeed.comm as dist
import os
from tqdm import tqdm
import copy
import random
import clip

from loss_func import CLIPLoss, StandardCLIPLoss, CombinedCLIPDPOLoss
from load_data import deduplicate_batch
from val import evaluate_cosine_similarities

def check_trainable_parameters(model):
    print("=== Parameter training flags ===")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")

def train_clip_model(
    model_engine, 
    data_loader, 
    sampler, 
    optimizer, 
    scheduler, 
    num_epochs, 
    device,
    learning_rate,
    mode='HNC', 
    lambda_ref = 0.01,
    hard_neg_weight = 1,
    dynamic_weight=True, 
    min_weight=0, 
    max_weight=10, 
    update_interval=1000,  
    num_updates=10,
    dpo_beta=1.0,  
    combined_alpha=0.5, 
    val_dataset=None, 
    val_step_frequency=100,
    checkpoint_dir=None    
    ):
    
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

    original_state_dict = copy.deepcopy(model_engine.state_dict())

    model_engine.train()

    # if dist.get_rank() == 0:
    #     check_trainable_parameters(model_engine)

    if mode.lower() == 'hnc':
        print('=======hnc mode=======')
        loss_fn = CLIPLoss(
            hard_neg_weight=hard_neg_weight, 
            lambda_reg=lambda_ref,
            dynamic_weight=dynamic_weight, 
            min_weight=min_weight, 
            max_weight=max_weight, 
            update_interval=update_interval,  
            num_updates=num_updates 
            )
    elif mode.lower() == 'standard':
        print('=======standard mode=======')
        loss_fn = StandardCLIPLoss()
    elif mode.lower() == 'dpo':
        print('======= DPO mode (Combined) =======')
        loss_fn = CombinedCLIPDPOLoss(lambda_reg=lambda_ref, beta=dpo_beta, alpha=combined_alpha)
    else:
        raise ValueError("mode must be either 'HNC', 'standard', or 'DPO'")

    global_step = 0 

    for epoch in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        if dist.get_rank() == 0:
            logging.info(f"Epoch {epoch + 1}/{num_epochs} begins.")
    
        total_loss = 0.0

        progress_bar = tqdm(
            data_loader,
            desc=f"Training Epoch {epoch + 1}/{num_epochs}",
            disable=(dist.get_rank() != 0),  
        )
    
        for step, batch in enumerate(progress_bar):
            if hasattr(loss_fn, "update_step"):
                loss_fn.update_step(global_step)

            images, text_inputs = deduplicate_batch(batch, device)

            with torch.cuda.amp.autocast():
                image_features = model_engine.module.encode_image(images)
                text_features = model_engine.module.encode_text(text_inputs)
                logit_scale = model_engine.module.logit_scale

            # Compute loss using our custom loss function.
            loss_outputs = loss_fn(
                image_features, 
                text_features, 
                logit_scale, 
                original_state_dict=original_state_dict, 
                model=model_engine
            )

            if mode.lower() == 'dpo':
                total_loss_val, contrastive_loss, dpo_loss, reg_loss = loss_outputs
                loss_dict = {
                    "loss_total": total_loss_val,
                    "loss_contrastive": contrastive_loss,
                    "loss_dpo": dpo_loss,
                    "reg_loss": reg_loss,
                }
            elif mode.lower() == 'standard':
                total_loss, loss_i2t, loss_t2i, positive_score, logit_scale = loss_outputs
                reg_loss = torch.tensor(0.0, device=loss_i2t.device)
                loss_dict = {
                    "loss_total": total_loss,
                    "loss_i2t": loss_i2t,
                    "loss_t2i": loss_t2i,
                    "loss_contrastive": (loss_i2t + loss_t2i) / 2.0,
                    "reg_loss": reg_loss,
                    "logit_scale": logit_scale.item(),
                    "positive_score": positive_score.mean().item(),
                }
            elif mode.lower() == 'hnc':
                total_loss, loss_i2t, loss_t2i, reg_loss, positive_score, hard_negative_scores, logit_scale  = loss_outputs
                loss_dict = {
                    "loss_total": total_loss,
                    "loss_i2t": loss_i2t,
                    "loss_t2i": loss_t2i,
                    "loss_contrastive": (loss_i2t + loss_t2i) / 2.0,
                    "reg_loss": reg_loss,
                    "logit_scale": logit_scale.item(),
                    "positive_score": positive_score.mean().item(),
                    'hard_negative_scores':hard_negative_scores.mean().item(),
                }
            else:
                print('There is no such mode')

            # Use the total loss for backpropagation.
            loss = loss_dict["loss_total"]

            # Backpropagation.
            optimizer.zero_grad()
            model_engine.backward(loss)
            model_engine.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            # Determine current weight for logging.
            if hasattr(loss_fn, "dynamic_weight") and loss_fn.dynamic_weight:
                max_update_step = loss_fn.update_interval * loss_fn.num_updates
                if loss_fn.current_step < max_update_step:
                    increment = (loss_fn.max_weight - loss_fn.min_weight) / loss_fn.num_updates
                    intervals_passed = loss_fn.current_step // loss_fn.update_interval
                    current_weight = loss_fn.min_weight + intervals_passed * increment
                else:
                    current_weight = loss_fn.max_weight
            elif hasattr(loss_fn, "hard_neg_weight"):
                current_weight = loss_fn.hard_neg_weight
            else:
                current_weight = 0 

            if dist.get_rank() == 0 and global_step % 100 == 0:
                wandb.log({
                    "loss_total": loss_dict["loss_total"].item(),
                    "loss_contrastive": loss_dict["loss_contrastive"].item(),
                    "loss_dpo": loss_dict.get("loss_dpo", torch.tensor(0.0)).item(),
                    "loss_i2t": loss_dict.get("loss_i2t", torch.tensor(0.0)).item(),
                    "loss_t2i": loss_dict.get("loss_t2i", torch.tensor(0.0)).item(),
                    "reg_loss": loss_dict["reg_loss"].item(),
                    "logit_scale": loss_dict["logit_scale"],
                    "positive_score": loss_dict.get("positive_score", 0.0),
                    "hard_negative_scores": loss_dict.get("hard_negative_scores", 0.0),
                    "current_lr": optimizer.param_groups[0]["lr"],
                    "global_step": global_step,
                    "hard_neg_weight": current_weight,
                })
            
            global_step += 1
 
            # ---- Begin Validation Step ----
            if val_dataset is not None and (global_step % val_step_frequency == 0):
                model_to_eval = model_engine.module if hasattr(model_engine, "module") else model_engine
                avg_pos, avg_neg, margin = evaluate_cosine_similarities(model_to_eval, val_loader, device)
                if dist.get_rank() == 0:
                    # print(f"Number of samples in val_dataset: {len(val_dataset)}")
                    # print(f"Number of samples in val_subset: {len(val_subset)}")
                    wandb.log({
                        "val/avg_pos": avg_pos,
                        "val/avg_neg": avg_neg,
                        "val/margin": margin,
                        "global_step": global_step,
                    })
            # ---- End Validation Step ----
            
                     
        avg_loss = total_loss / num_batches
        if dist.get_rank() == 0:
            logging.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss}")
            wandb.log({"epoch_loss": avg_loss})
        
        # if checkpoint_dir is not None:
        #     model_engine.save_checkpoint(checkpoint_dir, tag=f"epoch_{epoch+1}")
        
    
    wandb.finish()

    if checkpoint_dir is not None:
            model_engine.save_checkpoint(checkpoint_dir, tag='final')
        
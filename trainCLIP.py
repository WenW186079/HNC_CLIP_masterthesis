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
from eval.eval_functions import evaluate_cosine_similarities,evaluate_cosine_similarities_and_plot,evaluate_cosine_similarities_random_negtive

def set_trainable_parameters(model, finetune_mode):
    if finetune_mode == "text_encoder":
        # Freeze all visual encoder parameters.
        for param in model.visual.parameters():
            param.requires_grad = False
        for param in model.transformer.parameters():
            param.requires_grad = True
        if hasattr(model, 'text_projection'):
            if isinstance(model.text_projection, torch.nn.Parameter):
                model.text_projection.requires_grad = True
            else:
                for param in model.text_projection.parameters():
                    param.requires_grad = True
        if hasattr(model, 'token_embedding'):
            if isinstance(model.token_embedding, torch.nn.Parameter):
                model.token_embedding.requires_grad = True
            else:
                for param in model.token_embedding.parameters():
                    param.requires_grad = True

    elif finetune_mode == "vision_encoder":
        # Freeze the text encoder.
        for param in model.transformer.parameters():
            param.requires_grad = False
        if hasattr(model, 'text_projection'):
            if isinstance(model.text_projection, torch.nn.Parameter):
                model.text_projection.requires_grad = False
            else:
                for param in model.text_projection.parameters():
                    param.requires_grad = False

        if hasattr(model, 'token_embedding'):
            if isinstance(model.token_embedding, torch.nn.Parameter):
                model.token_embedding.requires_grad = False
            else:
                for param in model.token_embedding.parameters():
                    param.requires_grad = False
        # Ensure vision parameters are trainable.
        for param in model.visual.parameters():
            param.requires_grad = True

    elif finetune_mode == "full_encoder":
        # Train all parameters.
        for param in model.parameters():
            param.requires_grad = True

    elif finetune_mode == "full_encoder_last":
        # First, freeze all parameters.
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only the last block of the vision encoder.
        if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
            if hasattr(model.visual.transformer, 'resblocks'):
                # Unfreeze the parameters in the last residual block.
                for param in model.visual.transformer.resblocks[-1].parameters():
                    param.requires_grad = True
            else:
                # If the expected attribute doesn't exist, unfreeze all visual parameters.
                for param in model.visual.parameters():
                    param.requires_grad = True

        # Unfreeze only the last block of the text encoder.
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'resblocks'):
            for param in model.transformer.resblocks[-1].parameters():
                param.requires_grad = True
        else:
            # Fallback: unfreeze all text encoder parameters.
            for param in model.transformer.parameters():
                param.requires_grad = True

        # Unfreeze text_projection and token_embedding as well.
        if hasattr(model, 'text_projection'):
            if isinstance(model.text_projection, torch.nn.Parameter):
                model.text_projection.requires_grad = True
            else:
                for param in model.text_projection.parameters():
                    param.requires_grad = True
        if hasattr(model, 'token_embedding'):
            if isinstance(model.token_embedding, torch.nn.Parameter):
                model.token_embedding.requires_grad = True
            else:
                for param in model.token_embedding.parameters():
                    param.requires_grad = True

    else:
        print(f"Unknown finetune_mode: {finetune_mode}. No parameters were frozen.")
    
def check_trainable_parameters(model):
    print("=== Parameter training flags ===")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")


def save_checkpoint(model_engine, optimizer, epoch, checkpoint_dir, finetune_mode, filename=None):
    if hasattr(model_engine, "module"):
        state_dict = model_engine.module.state_dict()
    else:
        state_dict = model_engine.state_dict()

    checkpoint_dict = {
        "epoch": epoch,
        "state_dict": state_dict,
        "optimizer": optimizer.state_dict(),
        "finetune_mode": finetune_mode,
    }
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    if filename is None:
        filename = f"epoch_{epoch}_{finetune_mode}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint_dict, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

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
    val_loader=None,
    val_step_frequency=100,
    checkpoint_dir=None,
    finetune_mode=None 
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

    if dist.get_rank() == 0:
        check_trainable_parameters(model_engine)

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

            images, text_inputs = deduplicate_batch(batch, device, mode=mode)

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
                total_loss, contrastive_loss, dpo_loss, reg_loss = loss_outputs
                loss_dict = {
                    "loss_total": total_loss,
                    "contrastive_loss": contrastive_loss,
                    "dpo_loss": dpo_loss,
                    "reg_loss": reg_loss,
                }
            elif mode.lower() == 'standard':
                total_loss, loss_i2t, loss_t2i, positive_score, logit_scale = loss_outputs
                loss_dict = {
                    "loss_total": total_loss,
                    "loss_i2t": loss_i2t,
                    "loss_t2i": loss_t2i,
                    "loss_contrastive": total_loss,
                    "logit_scale": logit_scale,
                    "positive_score": positive_score.mean(),
                }
            elif mode.lower() == 'hnc':
                total_loss, loss_i2t, loss_t2i, reg_loss, positive_score, hard_negative_scores, logit_scale , margin = loss_outputs
                loss_dict = {
                    "loss_total": total_loss,
                    "loss_i2t": loss_i2t,
                    "loss_t2i": loss_t2i,
                    "loss_contrastive": (loss_i2t + loss_t2i) / 2.0,
                    "reg_loss": reg_loss,
                    "logit_scale": logit_scale,
                    "positive_score": positive_score.mean(),
                    "hard_negative_scores": hard_negative_scores.mean(),
                    "margin_in_train": margin,
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
            progress_bar.set_postfix()

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

            if dist.get_rank() == 0 and global_step % 10 == 0:
                if mode.lower() == 'dpo':
                    log_dict = {
                        "loss_total": loss_dict["loss_total"].item(),
                        "contrastive_loss": loss_dict["contrastive_loss"].item(),
                        "dpo_loss": loss_dict["dpo_loss"].item(),
                        "reg_loss": loss_dict["reg_loss"].item(),
                    }
                elif mode.lower() == 'standard':
                    log_dict = {
                        "loss_total": loss_dict["loss_total"].item(),
                        "loss_i2t": loss_dict["loss_i2t"].item(),
                        "loss_t2i": loss_dict["loss_t2i"].item(),
                        "loss_contrastive": loss_dict["loss_contrastive"].item(),
                        "logit_scale": loss_dict["logit_scale"].item() if isinstance(loss_dict["logit_scale"], torch.Tensor) else loss_dict["logit_scale"],
                        "positive_score": loss_dict["positive_score"].item() if isinstance(loss_dict["positive_score"], torch.Tensor) else loss_dict["positive_score"],
                    }
                elif mode.lower() == 'hnc':
                    log_dict = {
                        "loss_total": loss_dict["loss_total"].item(),
                        "loss_i2t": loss_dict["loss_i2t"].item(),
                        "loss_t2i": loss_dict["loss_t2i"].item(),
                        "loss_contrastive": loss_dict["loss_contrastive"].item(),
                        "reg_loss": loss_dict["reg_loss"].item(),
                        "logit_scale": loss_dict["logit_scale"].item() if isinstance(loss_dict["logit_scale"], torch.Tensor) else loss_dict["logit_scale"],
                        "positive_score": loss_dict["positive_score"].item() if isinstance(loss_dict["positive_score"], torch.Tensor) else loss_dict["positive_score"],
                        "hard_negative_scores": loss_dict["hard_negative_scores"].item() if isinstance(loss_dict["hard_negative_scores"], torch.Tensor) else loss_dict["hard_negative_scores"],
                        "margin_in_train": loss_dict["margin_in_train"].item() if isinstance(loss_dict["margin_in_train"], torch.Tensor) else loss_dict["margin_in_train"],
                    }
                else:
                    log_dict = {}
                
                log_dict["current_lr"] = optimizer.param_groups[0]["lr"]
                log_dict["global_step"] = global_step
                log_dict["hard_neg_weight"] = current_weight

                wandb.log(log_dict)
                        
            global_step += 1
 
            # ---- Begin Validation Step ----
            if val_loader is not None and (global_step % val_step_frequency == 0):
                model_to_eval = model_engine.module if hasattr(model_engine, "module") else model_engine
                # avg_pos, avg_neg, margin = evaluate_cosine_similarities(model_to_eval, val_loader, device)
                # if dist.get_rank() == 0:
                #     # print(f"Number of samples in val_dataset: {len(val_dataset)}")
                #     # print(f"Number of samples in val_subset: {len(val_subset)}")
                #     wandb.log({
                #         "val/avg_pos": avg_pos,
                #         "val/avg_neg": avg_neg,
                #         "val/margin": margin,
                #         "global_step": global_step,
                #     })
                avg_pos, avg_neg, avg_rand_neg, margin = evaluate_cosine_similarities_random_negtive(model_to_eval, val_loader, device)
                if dist.get_rank() == 0:
                    wandb.log({
                        "val/avg_pos": avg_pos,
                        "val/avg_neg": avg_neg,
                        "val/random_neg": avg_rand_neg,
                        "val/margin": margin,
                        "global_step": global_step,
                    })
            # ---- End Validation Step ----
            
                     
        avg_loss = total_loss / num_batches
        if dist.get_rank() == 0:
            logging.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss}")
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})
        
        # if checkpoint_dir is not None and dist.get_rank() == 0 and ((epoch + 1) % 5 == 0):
        if checkpoint_dir is not None and dist.get_rank() == 0 :
            save_checkpoint(model_engine, optimizer, epoch + 1, checkpoint_dir, finetune_mode, filename=None)
        
    wandb.finish()

    # After the training loop ends.
    if dist.get_rank() == 0 and checkpoint_dir is not None:
        filename = f"{finetune_mode}_final_model.pt"
        save_checkpoint(model_engine, optimizer, num_epochs, checkpoint_dir, finetune_mode, filename=filename)
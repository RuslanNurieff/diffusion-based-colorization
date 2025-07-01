# train.py

import logging
import math
import os

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (AutoencoderKL, DDPMScheduler,
                       StableDiffusionInstructPix2PixPipeline,
                       UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from PIL import Image
import requests # NEW: For downloading validation image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# Import local modules
from config import TrainingConfig
from data_utils import get_collate_fn, setup_dataset

# Setup logger
logger = get_logger(__name__, log_level="INFO")

# --- NEW: Helper function to log validation images ---
def log_validation_images(config, unet, vae, text_encoder, tokenizer, accelerator, weight_dtype, global_step):
    """Generates and logs validation images to W&B."""
    
    logger.info("Running validation... Generating images with validation prompt...")
    
    # Create pipeline
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        unet=accelerator.unwrap_model(unet),
        text_encoder=accelerator.unwrap_model(text_encoder),
        vae=accelerator.unwrap_model(vae),
        torch_dtype=weight_dtype,
    )
    pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    
    # Prepare validation inputs
    validation_image = Image.open(requests.get(config.validation_image_url, stream=True).raw).convert("RGB").resize((512, 512))
    
    # Generate images
    generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)
    val_images = []
    for _ in range(config.num_validation_images):
        with torch.autocast("cuda"):
            image = pipeline(
                prompt=config.validation_prompt, 
                image=validation_image, 
                generator=generator
            ).images[0]
            val_images.append(image)

    # Log to W&B
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            wandb_images = [
                tracker.Image(image, caption=f"Prompt: {config.validation_prompt}\nStep: {global_step}") 
                for image in val_images
            ]
            tracker.log({"validation_samples": wandb_images}, step=global_step)
            
    del pipeline
    torch.cuda.empty_cache()


def main():
    config = TrainingConfig()

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to, # MODIFIED: Use config for this
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Starting training with config: {config.__dict__}")

    if config.seed is not None:
        set_seed(config.seed)

    # --- NEW: Initialize W&B tracker ---
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        if config.report_to == "wandb":
            # The accelerator will automatically init a run from the config
            accelerator.init_trackers(config.wandb_project_name, config=vars(config))


    # Load models
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    train_dataset = setup_dataset(config, tokenizer, logger)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=get_collate_fn(),
        batch_size=config.train_batch_size, num_workers=config.dataloader_num_workers
    )
    
    optimizer_cls = torch.optim.AdamW
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            logger.warning("bitsandbytes not found, using regular AdamW.")

    optimizer = optimizer_cls(
        unet.parameters(), lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay, eps=config.adam_epsilon
    )
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        config.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=config.max_train_steps * config.gradient_accumulation_steps
    )
    
    unet, text_encoder, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, vae, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16
    
    # --- Start Training ---
    logger.info("***** Running training *****")
    progress_bar = tqdm(range(config.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0

    for epoch in range(config.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Training step logic (remains the same)
                latents = vae.encode(batch["edited_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                original_image_embeds = vae.encode(batch["original_pixel_values"].to(dtype=weight_dtype)).latent_dist.mode()
                model_pred = unet(torch.cat([noisy_latents, original_image_embeds], dim=1), timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": loss.detach().item()}, step=global_step)

                # --- NEW: Periodically log validation images ---
                if accelerator.is_main_process and (global_step % config.log_validation_images_every_n_steps == 0):
                    log_validation_images(config, unet, vae, text_encoder, tokenizer, accelerator, weight_dtype, global_step)


            if global_step >= config.max_train_steps:
                break
    
    # --- End of Training ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            config.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=accelerator.unwrap_model(vae),
        )
        pipeline.save_pretrained(config.output_dir)
        logger.info(f"Final model saved to {config.output_dir}")

    accelerator.end_training()

if __name__ == "__main__":
    main()
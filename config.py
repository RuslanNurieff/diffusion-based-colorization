# config.py

class TrainingConfig:
    """
    Configuration class for all training parameters.
    """
    # Model and data paths
    pretrained_model_name_or_path: str = "timbrooks/instruct-pix2pix"
    dataset_name: str = "annyorange/colorized-dataset"
    dataset_config_name: str = "full"
    output_dir: str = "instruct-pix2pix-colorizer-model"
    cache_dir: str = None

    # Dataset column names (customize for your dataset)
    original_image_column: str = "original_image"
    edited_image_column: str = "colorized_image"
    edit_prompt_column: str = "edit_prompt"
    
    # Training parameters
    seed: int = 42
    resolution: int = 256
    center_crop: bool = True
    random_flip: bool = True
    train_batch_size: int = 4
    num_train_epochs: int = 50
    max_train_steps: int = None
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-5
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    max_train_samples: int = None
    
    # Optimizer parameters
    use_8bit_adam: bool = True
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    
    # Accelerator and performance
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    allow_tf32: bool = True
    enable_xformers_memory_efficient_attention: bool = True
    dataloader_num_workers: int = 4
    
    # --- NEW: W&B and Validation Logging ---
    report_to: str = "wandb"  # Set to "wandb" to enable logging
    wandb_project_name: str = "instruct_pix2pix_finetune"
    
    # How often to log validation images to W&B
    log_validation_images_every_n_steps: int = 250
    
    # Validation prompt and image for logging
    validation_prompt: str = "make it look like a watercolor painting"
    validation_image_url: str = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_2.png"
    num_validation_images: int = 4
import torch
from datasets import load_dataset
from torchvision import transforms
from transformers import CLIPTokenizer

def get_train_transforms(config):
    """Returns the image transformations for training."""
    return transforms.Compose([
        transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(config.resolution) if config.center_crop else transforms.RandomCrop(config.resolution),
        transforms.RandomHorizontalFlip() if config.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

def setup_dataset(config, tokenizer, logger):
    """Loads and preprocesses the dataset."""
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config_name,
        cache_dir=config.cache_dir,
    )
    
    train_transforms = get_train_transforms(config)

    def tokenize_captions(captions):
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    def preprocess_train(examples):
        original_images = [image.convert("RGB") for image in examples[config.original_image_column]]
        edited_images = [image.convert("RGB") for image in examples[config.edited_image_column]]
        
        examples["original_pixels"] = [train_transforms(image) for image in original_images]
        examples["modified_pixels"] = [train_transforms(image) for image in edited_images]

        captions = [caption for caption in examples[config.edit_prompt_column]]
        examples["input_tokens"] = tokenize_captions(captions)
        return examples

    if config.max_train_samples:
        logger.info(f"Truncating training set to {config.max_train_samples} samples.")
        dataset["train"] = dataset["train"].shuffle(seed=config.seed).select(range(config.max_train_samples))
        
    train_dataset = dataset["train"].with_transform(preprocess_train)
    return train_dataset

def get_collate_fn():
    """Returns the collate function for the dataloader."""
    def collate_fn(examples):
        original_pixel_values = torch.stack([example["original_pixels"] for example in examples])
        edited_pixel_values = torch.stack([example["modified_pixels"] for example in examples])
        input_ids = torch.stack([example["input_tokens"] for example in examples])
        
        return {
            "original_pixels": original_pixel_values.to(memory_format=torch.contiguous_format).float(),
            "edited_pixels": edited_pixel_values.to(memory_format=torch.contiguous_format).float(),
            "input_tokens": input_ids,
        }
    return collate_fn
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import requests

def download_image(url: str) -> Image.Image:
    """Downloads an image from a URL and returns a PIL Image."""
    image = Image.open(requests.get(url, stream=True).raw)
    image = Image.ImageOps.exif_transpose(image)
    return image.convert("RGB")

def run_inference(
    model_path: str,
    image_url: str,
    instruction: str,
    output_path: str = "edited_image.png",
    image_guidance_scale: float = 1.5,
    guidance_scale: float = 7.0,
):
    """
    Runs inference using a fine-tuned InstructPix2Pix model.

    Args:
        model_path (str): Path to the fine-tuned pipeline directory.
        image_url (str): URL of the image to edit.
        instruction (str): The instruction for editing the image.
        output_path (str): Path to save the edited image.
        image_guidance_scale (float): The image guidance scale.
        guidance_scale (float): The text guidance scale.
    """
    print(f"Loading model from: {model_path}")
    # Load the fine-tuned pipeline
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe = pipe.to("cuda")

    print(f"Downloading image from: {image_url}")
    original_image = download_image(image_url).resize((512, 512))

    print(f"Running inference with instruction: '{instruction}'")
    edited_image = pipe(
        prompt=instruction,
        image=original_image,
        num_inference_steps=20,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
    ).images[0]

    edited_image.save(output_path)
    print(f"Successfully saved edited image to: {output_path}")


if __name__ == "__main__":
    # Import the config to get the output directory of the trained model
    from config import TrainingConfig
    
    config = TrainingConfig()

    # --- Example Usage ---
    # Ensure you have trained a model and it's located in the output_dir.
    fine_tuned_model_path = config.output_dir

    # URL of a black and white image
    input_image_url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_2.png"
    
    # The instruction you want to apply
    edit_instruction = "add color to the image"

    run_inference(
        model_path=fine_tuned_model_path,
        image_url=input_image_url,
        instruction=edit_instruction,
    )
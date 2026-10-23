from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
from pathlib import Path
import pandas as pd


class BlipCaptioner:
    def __init__(
        self, model_name: str = "Salesforce/blip2-opt-2.7b-coco", max_length: int = 75
    ):
        """Initialize BLIP-2 model for batch captioning."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.max_length = max_length

    def generate_caption(self, image: Image.Image) -> str:
        """Generate caption for a single image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_length=self.max_length)
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ].strip()
        return caption

    def process_images(self, image_paths) -> dict:
        """Process multiple images and return a dictionary of their captions."""
        captions = {}
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            caption = self.generate_caption(img)
            captions[str(path)] = caption
        return captions

    def process_and_save(self, image_paths, output_dir: Path) -> dict:
        """Process images, save captions to CSV, and return caption dictionary."""
        captions = self.process_images(image_paths)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(
            [{"image_path": k, "caption": v} for k, v in captions.items()]
        )
        df.to_csv(output_dir / "blip_captions.csv", index=False)

        return captions

    def __del__(self):
        """Cleanup any GPU memory."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

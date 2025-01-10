import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipForConditionalGeneration, BlipProcessor

# Load CLIP model for feature extraction
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load BLIP model for captioning
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Example Image
image = Image.open("example_image.jpg")

# Feature extraction using CLIP
inputs = clip_processor(images=image, return_tensors="pt")
clip_features = clip_model.get_image_features(**inputs)

# Simulated retrieval (in practice, search in vector database)
retrieved_features = clip_features  # Replace with actual retrieval

# Generate caption with BLIP
caption_inputs = blip_processor(images=image, return_tensors="pt")
generated_ids = blip_model.generate(**caption_inputs)
caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)

print(f"Generated Caption: {caption}")

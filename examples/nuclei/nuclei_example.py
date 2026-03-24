#!/usr/bin/env python3

import torch
from datasets import load_dataset
from torchvision.models import MobileNet_V2_Weights

# Test model. The output is a likeness-score for each class in ImageNet, pick top-1 for result
def print_result(output_data, true_label):
    imagenet_labels = MobileNet_V2_Weights.DEFAULT.meta["categories"]
    output_data = torch.Tensor(output_data)
    index = output_data.argmax().item()
    predicted_label = imagenet_labels[index]
    print(f"True label: {true_label}. Model output: {predicted_label}")
    
    return true_label == predicted_label

# Generator yielding (transformed_img, label_name, original_img)
def sample_generator(max_samples=200):
    # Load a small ImageNet validation split from Hugging Face
    dataset = load_dataset("frgfm/imagenette", "full_size", split="validation")
    
    # Shuffle deterministically
    dataset = dataset.shuffle(seed=0)
    
    # MobileNetV2 preprocessing transforms
    weights = MobileNet_V2_Weights.DEFAULT

    preprocess = weights.transforms()

    label_names = dataset.features["label"].names
    for i, sample in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        image = sample["image"].convert("RGB") 
        yield (
            preprocess(image).unsqueeze(0).to(memory_format=torch.channels_last),
            label_names[sample["label"]],
            sample["image"],
        )

# Display sample image before and after transform
transformed_img, label, original_img = next(sample_generator())

from torchvision.models import mobilenet_v2

if __name__ == "__main__":
    # Init model
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model = model.eval()
    
    output_data = model(transformed_img)
    print_result(output_data, label)


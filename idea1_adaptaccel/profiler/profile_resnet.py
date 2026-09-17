"""
profile_resnet.py — Profile ResNet50 on ImageNet-like data.

Profiles dense + pruned (50%, 70%, 90% sparsity) versions.
Uses random input data (same dimensions as ImageNet: 3×224×224).

Usage:
    cd project/idea1_adaptaccel
    python -m profiler.profile_resnet
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from profiler.workload_profiler import LayerProfiler

N_SAMPLES = 200  # Number of input images to profile
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "profiling_data")


def create_dummy_imagenet_loader(n_samples, batch_size=16):
    """Create a DataLoader with random ImageNet-sized inputs."""
    images = torch.randn(n_samples, 3, 224, 224)
    labels = torch.randint(0, 1000, (n_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def apply_unstructured_pruning(model, sparsity):
    """Apply global unstructured L1 pruning to all Conv2d and Linear layers."""
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    if len(parameters_to_prune) == 0:
        return model

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )

    # Make pruning permanent
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

    return model


def profile_one_config(model_name, model, loader, output_dir):
    """Profile a single model configuration."""
    print(f"\n{'='*60}")
    print(f"  Profiling: {model_name}")
    print(f"{'='*60}")

    profiler = LayerProfiler(model, model_name=model_name)
    df = profiler.profile(loader, n_samples=N_SAMPLES)
    profiler.save_traces(output_dir)
    
    # Print summary
    print(f"\n  Summary for {model_name}:")
    print(f"  Layers profiled: {df['layer_name'].nunique()}")
    print(f"  Avg input sparsity:  {df['input_sparsity'].mean():.3f}")
    print(f"  Avg output sparsity: {df['output_sparsity'].mean():.3f}")
    print(f"  Avg weight sparsity: {df['weight_sparsity'].mean():.3f}")
    return df


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    loader = create_dummy_imagenet_loader(N_SAMPLES)

    # 1) Dense ResNet50
    model_dense = models.resnet50(weights=None).to(device)
    model_dense.eval()
    profile_one_config("resnet50_dense", model_dense, loader, OUTPUT_DIR)

    # 2) Pruned at 50%
    model_50 = models.resnet50(weights=None).to(device)
    apply_unstructured_pruning(model_50, sparsity=0.50)
    model_50.eval()
    profile_one_config("resnet50_pruned50", model_50, loader, OUTPUT_DIR)

    # 3) Pruned at 70%
    model_70 = models.resnet50(weights=None).to(device)
    apply_unstructured_pruning(model_70, sparsity=0.70)
    model_70.eval()
    profile_one_config("resnet50_pruned70", model_70, loader, OUTPUT_DIR)

    # 4) Pruned at 90%
    model_90 = models.resnet50(weights=None).to(device)
    apply_unstructured_pruning(model_90, sparsity=0.90)
    model_90.eval()
    profile_one_config("resnet50_pruned90", model_90, loader, OUTPUT_DIR)

    print(f"\n✓ All ResNet50 profiles saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

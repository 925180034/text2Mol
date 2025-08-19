#!/usr/bin/env python3
"""
Check checkpoint structure and keys.
"""

import torch
import sys

checkpoint_path = "/root/autodl-tmp/text2Mol-stage2/best_model_stage2.pt"

print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("\nCheckpoint keys:")
for key in checkpoint.keys():
    print(f"  - {key}")

if 'model_state_dict' in checkpoint:
    print("\nModel state dict keys (first 20):")
    keys = list(checkpoint['model_state_dict'].keys())
    for i, key in enumerate(keys[:20]):
        print(f"  {i+1}. {key}")
    print(f"  ... Total: {len(keys)} keys")
    
if 'state_dict' in checkpoint:
    print("\nState dict keys (first 20):")
    keys = list(checkpoint['state_dict'].keys())
    for i, key in enumerate(keys[:20]):
        print(f"  {i+1}. {key}")
    print(f"  ... Total: {len(keys)} keys")

# Check if it's a raw state dict
if not isinstance(checkpoint, dict) or ('model_state_dict' not in checkpoint and 'state_dict' not in checkpoint):
    # It might be a raw state dict
    print("\nRaw state dict keys (first 20):")
    keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    for i, key in enumerate(keys[:20]):
        print(f"  {i+1}. {key}")
    print(f"  ... Total: {len(keys)} keys")
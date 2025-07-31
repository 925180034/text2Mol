# Model Setup Instructions

The pre-trained models are not included in this repository due to their large size. Please follow these steps to download the required models.

## Required Models

### MolT5-Small (Recommended for testing)
- **Size**: ~300MB
- **Repository**: `laituan245/molt5-small`
- **Local Path**: `models/MolT5-Small/`

### Alternative Models
- **MolT5-Base**: `laituan245/molt5-base` (~1GB)
- **MolT5-Large**: `laituan245/molt5-large` (~3GB)

## Quick Setup

### Method 1: Using Transformers (Recommended)
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

# Set up Hugging Face mirror (optional, for faster download in China)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Download model
model_name = "laituan245/molt5-small"
local_path = "models/MolT5-Small"

# Create directory
os.makedirs(local_path, exist_ok=True)

# Download and save
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

tokenizer.save_pretrained(local_path)
model.save_pretrained(local_path)

print(f"✅ Model saved to {local_path}")
```

### Method 2: Using Hugging Face Hub
```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download using Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='laituan245/molt5-small', local_dir='models/MolT5-Small')
print('✅ Model downloaded successfully')
"
```

### Method 3: Using Git LFS
```bash
# Make sure git-lfs is installed
git lfs install

# Clone the model repository
git clone https://huggingface.co/laituan245/molt5-small models/MolT5-Small
```

## Verification

After downloading, verify the model setup:

```python
# Run the system check
python check_system.py
```

The model directory should contain:
- `config.json`
- `model.safetensors` (or `pytorch_model.bin`)
- `tokenizer_config.json`
- `special_tokens_map.json`
- `spiece.model`

## Configuration

Update the model path in your configuration file if using a different model:

```yaml
# configs/default_config.yaml
model:
  molt5_checkpoint: "models/MolT5-Small"  # or your chosen model path
```

## Troubleshooting

### Common Issues

1. **Network Issues**: Use HF_ENDPOINT mirror for faster downloads
2. **Disk Space**: Ensure you have enough space (300MB+ for small model)
3. **Dependencies**: Make sure `transformers` and `torch` are installed

### Alternative Download Sources

If Hugging Face is not accessible, you can:
1. Download models manually from mirrors
2. Use academic network resources
3. Contact repository maintainer for alternative sources

## Model Comparison

| Model | Size | Parameters | Use Case |
|-------|------|------------|----------|
| MolT5-Small | ~300MB | 77M | Testing, development |
| MolT5-Base | ~1GB | 250M | Balanced performance |
| MolT5-Large | ~3GB | 770M | Best performance |

For most use cases, **MolT5-Small** provides a good balance of performance and resource usage.
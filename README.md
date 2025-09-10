# Generative AI Projects - Asset Management

This repository contains multiple Generative AI projects with centralized asset management.

## Project Structure

```
Generative AI/
├── asset_manager.py          # Centralized asset management system
├── setup_assets.py           # Asset setup script
├── Datasets/                 # Centralized datasets directory
│   ├── certificates/         # ComfyUI certificate assets
│   ├── comfyui_models/       # ComfyUI model files
│   ├── comfyui_outputs/      # ComfyUI output files
│   ├── comfyui_configs/      # ComfyUI configuration files
│   ├── neural_networks/      # Neural network datasets and results
│   ├── safety_chatbot/       # Safety chatbot datasets and models
│   └── transformers/         # Transformer models and results
├── ComfyUI/                  # ComfyUI project
├── Neural Networks/          # Neural network examples
├── Safety_chatbot/           # Industrial safety chatbot
└── Transformers/             # Transformer model examples
```

## Asset Management

The `AssetManager` class provides centralized asset management across all projects:

### Features:
- Cross-platform path handling using `os.path`
- Centralized dataset storage
- Project-specific asset organization
- Asset registry and metadata tracking
- Automatic directory creation

### Usage:

```python
from asset_manager import AssetManager

# Initialize asset manager
am = AssetManager()

# Get asset path
certificate_path = am.get_asset_path('comfyui', 'certificates', 'Certificate.jpeg')

# Register new asset
am.register_asset('comfyui', 'certificates', 'new_certificate.jpg', 
                 certificate_path, {'description': 'New certificate'})

# List all assets
all_assets = am.list_assets()
```

## Projects

### 1. ComfyUI
- **Purpose**: AI image generation and processing
- **Assets**: Certificates, models, outputs, configurations
- **Main Script**: `ComfyUI/certificate_text_overlay/adding_name.py`

### 2. Neural Networks
- **Purpose**: CNN vs Transformer comparison and RNN text processing
- **Assets**: Datasets, models, results, plots
- **Main Scripts**: 
  - `Neural Networks/cnn_example/cnn_vs_transformer_demo.py`
  - `Neural Networks/rnn_example/rnn_demo.py`

### 3. Safety Chatbot
- **Purpose**: Industrial safety incident classification
- **Assets**: Datasets, models, embeddings, logs
- **Main Script**: `Safety_chatbot/safety_chatbot.py`

### 4. Transformers
- **Purpose**: BERT tokenization and analysis
- **Assets**: Datasets, models, tokenizers, results
- **Main Script**: `Transformers/Bert/bert_tokenization_demo.py`

## Getting Started

1. Run the setup script to initialize assets:
   ```bash
   python setup_assets.py
   ```

2. Each project has an `asset_links.py` file for easy asset access:
   ```python
   from asset_links import get_asset_path, list_assets
   
   # Get path to a specific asset
   path = get_asset_path('certificates', 'Certificate.jpeg')
   
   # List all assets of a type
   assets = list_assets('certificates')
   ```

## Asset Types by Project

### ComfyUI
- `certificates`: Certificate images and processed versions
- `models`: AI models and checkpoints
- `outputs`: Generated images and results
- `configs`: Configuration files

### Neural Networks
- `datasets`: Training and test datasets
- `models`: Trained model files
- `results`: Analysis results and metrics
- `plots`: Visualization plots and charts

### Safety Chatbot
- `datasets`: Safety incident datasets
- `models`: Trained classification models
- `embeddings`: Precomputed embeddings
- `logs`: Training and inference logs

### Transformers
- `datasets`: Text datasets for training
- `models`: Pre-trained transformer models
- `tokenizers`: Tokenizer configurations
- `results`: Analysis results and visualizations

## File Path Handling

All file paths are handled using the `os` module for cross-platform compatibility:
- Uses `os.path.join()` for path construction
- Uses `os.path.abspath()` for absolute paths
- Automatically creates directories as needed
- Handles both relative and absolute paths

## Asset Registry

The asset registry (`Datasets/asset_registry.json`) tracks all registered assets with metadata:
- Asset location and type
- Registration timestamp
- Custom metadata
- Project association

This ensures easy discovery and management of all project assets.

"""
Asset Links for Transformers Project
========================================

This file provides easy access to all assets for the transformers project.
Generated automatically by AssetManager.
"""

import os

# Base paths
BASE_PATH = r"/Users/vignesh/Documents/GitHub/Generative AI"
DATASETS_PATH = r"/Users/vignesh/Documents/GitHub/Generative AI/Datasets"

# Project-specific asset paths
ASSETS = {
    "datasets": r"/Users/vignesh/Documents/GitHub/Generative AI/Datasets/transformers",
    "models": r"/Users/vignesh/Documents/GitHub/Generative AI/Datasets/transformers/models",
    "tokenizers": r"/Users/vignesh/Documents/GitHub/Generative AI/Datasets/transformers/tokenizers",
    "results": r"/Users/vignesh/Documents/GitHub/Generative AI/Datasets/transformers/results",
}

def get_asset_path(asset_type: str, asset_name: str = None) -> str:
    """
    Get the path to an asset
    
    Args:
        asset_type: Type of asset (certificates, models, datasets, etc.)
        asset_name: Name of the asset file (optional)
    
    Returns:
        Path to the asset directory or specific file
    """
    if asset_type not in ASSETS:
        raise ValueError(f"Unknown asset type: {asset_type}. Available: {list(ASSETS.keys())}")
    
    base_path = ASSETS[asset_type]
    
    if asset_name:
        return os.path.join(base_path, asset_name)
    else:
        return base_path

def list_assets(asset_type: str) -> list:
    """
    List all assets of a specific type
    
    Args:
        asset_type: Type of assets to list
    
    Returns:
        List of asset names
    """
    asset_dir = get_asset_path(asset_type)
    if os.path.exists(asset_dir):
        return [f for f in os.listdir(asset_dir) if os.path.isfile(os.path.join(asset_dir, f))]
    return []

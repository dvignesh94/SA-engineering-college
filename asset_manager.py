"""
Centralized Asset Manager for Generative AI Projects
==================================================

This module provides a centralized way to manage assets and datasets across all
Generative AI projects. It uses the os module for proper path handling and
ensures all assets are organized under the centralized Datasets directory.

Features:
- Cross-platform path handling using os.path
- Centralized dataset management
- Project-specific asset organization
- Easy asset discovery and access
- Automatic directory creation
"""

import os
import json
from typing import Dict, List, Optional, Union
from pathlib import Path
import shutil

class AssetManager:
    """Centralized asset manager for all Generative AI projects"""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the AssetManager
        
        Args:
            base_path: Base path for the Generative AI workspace. 
                      If None, uses the current working directory.
        """
        if base_path is None:
            # Get the absolute path of the current script's directory
            self.base_path = os.path.dirname(os.path.abspath(__file__))
        else:
            self.base_path = os.path.abspath(base_path)
        
        # Define project structure
        self.projects = {
            'comfyui': 'ComfyUI',
            'neural_networks': 'Neural Networks',
            'safety_chatbot': 'Safety_chatbot',
            'transformers': 'Transformers'
        }
        
        # Centralized datasets directory
        self.datasets_dir = os.path.join(self.base_path, 'Datasets')
        
        # Project-specific asset directories
        self.asset_dirs = {
            'comfyui': {
                'certificates': os.path.join(self.datasets_dir, 'certificates'),
                'models': os.path.join(self.datasets_dir, 'comfyui_models'),
                'outputs': os.path.join(self.datasets_dir, 'comfyui_outputs'),
                'configs': os.path.join(self.datasets_dir, 'comfyui_configs')
            },
            'neural_networks': {
                'datasets': os.path.join(self.datasets_dir, 'neural_networks'),
                'models': os.path.join(self.datasets_dir, 'neural_networks', 'models'),
                'results': os.path.join(self.datasets_dir, 'neural_networks', 'results'),
                'plots': os.path.join(self.datasets_dir, 'neural_networks', 'plots')
            },
            'safety_chatbot': {
                'datasets': os.path.join(self.datasets_dir, 'safety_chatbot'),
                'models': os.path.join(self.datasets_dir, 'safety_chatbot', 'models'),
                'embeddings': os.path.join(self.datasets_dir, 'safety_chatbot', 'embeddings'),
                'logs': os.path.join(self.datasets_dir, 'safety_chatbot', 'logs')
            },
            'transformers': {
                'datasets': os.path.join(self.datasets_dir, 'transformers'),
                'models': os.path.join(self.datasets_dir, 'transformers', 'models'),
                'tokenizers': os.path.join(self.datasets_dir, 'transformers', 'tokenizers'),
                'results': os.path.join(self.datasets_dir, 'transformers', 'results')
            }
        }
        
        # Initialize directories
        self._create_directories()
        
        # Load asset registry
        self.registry_file = os.path.join(self.datasets_dir, 'asset_registry.json')
        self.registry = self._load_registry()
    
    def _create_directories(self):
        """Create all necessary directories"""
        # Create main datasets directory
        os.makedirs(self.datasets_dir, exist_ok=True)
        
        # Create project-specific directories
        for project, dirs in self.asset_dirs.items():
            for dir_name, dir_path in dirs.items():
                os.makedirs(dir_path, exist_ok=True)
        
        print(f"✅ Asset directories initialized at: {self.datasets_dir}")
    
    def _load_registry(self) -> Dict:
        """Load the asset registry from JSON file"""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️ Warning: Could not load asset registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save the asset registry to JSON file"""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except IOError as e:
            print(f"❌ Error saving asset registry: {e}")
    
    def register_asset(self, project: str, asset_type: str, asset_name: str, 
                      file_path: str, metadata: Optional[Dict] = None) -> str:
        """
        Register an asset in the registry
        
        Args:
            project: Project name (comfyui, neural_networks, safety_chatbot, transformers)
            asset_type: Type of asset (certificates, models, datasets, etc.)
            asset_name: Name of the asset
            file_path: Path to the asset file
            metadata: Optional metadata about the asset
            
        Returns:
            The registered asset path
        """
        if project not in self.projects:
            raise ValueError(f"Unknown project: {project}. Available: {list(self.projects.keys())}")
        
        if asset_type not in self.asset_dirs[project]:
            raise ValueError(f"Unknown asset type for {project}: {asset_type}. Available: {list(self.asset_dirs[project].keys())}")
        
        # Get the target directory for this asset type
        target_dir = self.asset_dirs[project][asset_type]
        
        # Create full path for the asset
        asset_path = os.path.join(target_dir, asset_name)
        
        # Register in the registry
        if project not in self.registry:
            self.registry[project] = {}
        
        if asset_type not in self.registry[project]:
            self.registry[project][asset_type] = {}
        
        self.registry[project][asset_type][asset_name] = {
            'path': asset_path,
            'metadata': metadata or {},
            'registered_at': str(Path().cwd())
        }
        
        # Save registry
        self._save_registry()
        
        return asset_path
    
    def get_asset_path(self, project: str, asset_type: str, asset_name: str) -> str:
        """
        Get the path to a registered asset
        
        Args:
            project: Project name
            asset_type: Type of asset
            asset_name: Name of the asset
            
        Returns:
            Path to the asset
        """
        try:
            return self.registry[project][asset_type][asset_name]['path']
        except KeyError:
            # If not registered, construct the path
            target_dir = self.asset_dirs[project][asset_type]
            return os.path.join(target_dir, asset_name)
    
    def list_assets(self, project: Optional[str] = None, asset_type: Optional[str] = None) -> Dict:
        """
        List all registered assets
        
        Args:
            project: Filter by project (optional)
            asset_type: Filter by asset type (optional)
            
        Returns:
            Dictionary of assets
        """
        if project and asset_type:
            return self.registry.get(project, {}).get(asset_type, {})
        elif project:
            return self.registry.get(project, {})
        else:
            return self.registry
    
    def copy_asset(self, source_path: str, project: str, asset_type: str, 
                   asset_name: Optional[str] = None) -> str:
        """
        Copy an asset to the centralized location
        
        Args:
            source_path: Path to the source file
            project: Target project
            asset_type: Target asset type
            asset_name: New name for the asset (optional)
            
        Returns:
            Path to the copied asset
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        # Get target directory
        target_dir = self.asset_dirs[project][asset_type]
        
        # Determine asset name
        if asset_name is None:
            asset_name = os.path.basename(source_path)
        
        # Full target path
        target_path = os.path.join(target_dir, asset_name)
        
        # Copy the file
        shutil.copy2(source_path, target_path)
        
        # Register the asset
        self.register_asset(project, asset_type, asset_name, target_path)
        
        print(f"✅ Copied {source_path} to {target_path}")
        return target_path
    
    def get_project_paths(self, project: str) -> Dict[str, str]:
        """
        Get all paths for a specific project
        
        Args:
            project: Project name
            
        Returns:
            Dictionary of project paths
        """
        if project not in self.projects:
            raise ValueError(f"Unknown project: {project}")
        
        return {
            'project_root': os.path.join(self.base_path, self.projects[project]),
            'assets': self.asset_dirs[project]
        }
    
    def setup_project_links(self, project: str) -> Dict[str, str]:
        """
        Setup symbolic links or references for a project
        
        Args:
            project: Project name
            
        Returns:
            Dictionary of setup information
        """
        project_paths = self.get_project_paths(project)
        
        # Create a links file for easy access
        links_file = os.path.join(project_paths['project_root'], 'asset_links.py')
        
        links_content = f'''"""
Asset Links for {project.title()} Project
========================================

This file provides easy access to all assets for the {project} project.
Generated automatically by AssetManager.
"""

import os

# Base paths
BASE_PATH = r"{self.base_path}"
DATASETS_PATH = r"{self.datasets_dir}"

# Project-specific asset paths
ASSETS = {{
'''
        
        for asset_type, path in self.asset_dirs[project].items():
            links_content += f'    "{asset_type}": r"{path}",\n'
        
        links_content += '''}

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
'''
        
        # Write the links file
        with open(links_file, 'w') as f:
            f.write(links_content)
        
        print(f"✅ Created asset links file: {links_file}")
        
        return {
            'links_file': links_file,
            'assets': self.asset_dirs[project]
        }

def main():
    """Main function to demonstrate AssetManager usage"""
    print("🚀 Initializing Asset Manager...")
    
    # Initialize the asset manager
    am = AssetManager()
    
    print(f"📁 Base path: {am.base_path}")
    print(f"📁 Datasets directory: {am.datasets_dir}")
    
    # Setup links for all projects
    for project in am.projects.keys():
        print(f"\n🔗 Setting up links for {project}...")
        am.setup_project_links(project)
    
    print("\n✅ Asset Manager setup complete!")
    print("\n📋 Available projects and asset types:")
    for project, asset_types in am.asset_dirs.items():
        print(f"  {project}: {list(asset_types.keys())}")

if __name__ == "__main__":
    main()

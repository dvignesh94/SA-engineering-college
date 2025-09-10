"""
Asset Management Demo
====================

This script demonstrates the centralized asset management system
across all Generative AI projects.
"""

import os
from asset_manager import AssetManager

def demo_asset_management():
    """Demonstrate the asset management system"""
    
    print("🎯 ASSET MANAGEMENT DEMONSTRATION")
    print("=" * 50)
    
    # Initialize asset manager
    am = AssetManager()
    
    print(f"📁 Base path: {am.base_path}")
    print(f"📁 Datasets directory: {am.datasets_dir}")
    
    # Show project structure
    print("\n📋 PROJECT STRUCTURE:")
    for project, project_name in am.projects.items():
        print(f"\n  {project.upper()} ({project_name}):")
        project_paths = am.get_project_paths(project)
        print(f"    Project root: {project_paths['project_root']}")
        print(f"    Asset directories:")
        for asset_type, path in project_paths['assets'].items():
            print(f"      {asset_type}: {path}")
    
    # List all registered assets
    print("\n📊 REGISTERED ASSETS:")
    registry = am.list_assets()
    
    total_assets = 0
    for project, asset_types in registry.items():
        print(f"\n  {project.upper()}:")
        for asset_type, assets in asset_types.items():
            print(f"    {asset_type}: {len(assets)} assets")
            total_assets += len(assets)
            for asset_name, asset_info in assets.items():
                print(f"      - {asset_name}")
                if asset_info.get('metadata'):
                    print(f"        Metadata: {asset_info['metadata']}")
    
    print(f"\n📈 Total registered assets: {total_assets}")
    
    # Demonstrate asset path retrieval
    print("\n🔍 ASSET PATH EXAMPLES:")
    
    # ComfyUI certificate
    cert_path = am.get_asset_path('comfyui', 'certificates', 'Certificate.jpeg')
    print(f"  ComfyUI Certificate: {cert_path}")
    print(f"    Exists: {os.path.exists(cert_path)}")
    
    # Safety chatbot dataset
    safety_path = am.get_asset_path('safety_chatbot', 'datasets', 
                                   'Industrial_safety_and_health_database_with_accidents_description.csv')
    print(f"  Safety Dataset: {safety_path}")
    print(f"    Exists: {os.path.exists(safety_path)}")
    
    # Neural networks dataset
    iris_path = am.get_asset_path('neural_networks', 'datasets', 'Iris.csv')
    print(f"  Iris Dataset: {iris_path}")
    print(f"    Exists: {os.path.exists(iris_path)}")
    
    # Demonstrate asset links files
    print("\n🔗 ASSET LINKS FILES:")
    for project in am.projects.keys():
        project_paths = am.get_project_paths(project)
        links_file = os.path.join(project_paths['project_root'], 'asset_links.py')
        if os.path.exists(links_file):
            print(f"  ✅ {project}: {links_file}")
        else:
            print(f"  ❌ {project}: Links file not found")
    
    # Show directory structure
    print("\n📁 DIRECTORY STRUCTURE:")
    def show_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        if os.path.isdir(path):
            items = sorted(os.listdir(path))
            for i, item in enumerate(items):
                if item.startswith('.'):
                    continue
                    
                item_path = os.path.join(path, item)
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{item}")
                
                if os.path.isdir(item_path) and current_depth < max_depth - 1:
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    show_tree(item_path, next_prefix, max_depth, current_depth + 1)
    
    show_tree(am.datasets_dir, max_depth=2)
    
    print("\n✅ Asset management demonstration complete!")

def demo_project_usage():
    """Demonstrate how projects use the asset management system"""
    
    print("\n🎮 PROJECT USAGE EXAMPLES")
    print("=" * 50)
    
    # Example 1: ComfyUI certificate processing
    print("\n1. ComfyUI Certificate Processing:")
    print("   ```python")
    print("   from asset_manager import AssetManager")
    print("   am = AssetManager()")
    print("   certificate_path = am.get_asset_path('comfyui', 'certificates', 'Certificate.jpeg')")
    print("   output_path = am.get_asset_path('comfyui', 'certificates', 'processed_certificate.jpg')")
    print("   # Process certificate...")
    print("   am.register_asset('comfyui', 'certificates', 'processed_certificate.jpg', output_path)")
    print("   ```")
    
    # Example 2: Neural Networks model saving
    print("\n2. Neural Networks Model Saving:")
    print("   ```python")
    print("   am = AssetManager()")
    print("   model_path = am.get_asset_path('neural_networks', 'models', 'cnn_model.pth')")
    print("   plot_path = am.get_asset_path('neural_networks', 'plots', 'training_curve.png')")
    print("   # Save model and plot...")
    print("   am.register_asset('neural_networks', 'models', 'cnn_model.pth', model_path)")
    print("   am.register_asset('neural_networks', 'plots', 'training_curve.png', plot_path)")
    print("   ```")
    
    # Example 3: Safety Chatbot embeddings
    print("\n3. Safety Chatbot Embeddings:")
    print("   ```python")
    print("   am = AssetManager()")
    print("   dataset_path = am.get_asset_path('safety_chatbot', 'datasets', 'safety_data.csv')")
    print("   embeddings_path = am.get_asset_path('safety_chatbot', 'embeddings', 'embeddings.npy')")
    print("   # Load data, create embeddings...")
    print("   am.register_asset('safety_chatbot', 'embeddings', 'embeddings.npy', embeddings_path)")
    print("   ```")
    
    # Example 4: Transformers results
    print("\n4. Transformers Analysis Results:")
    print("   ```python")
    print("   am = AssetManager()")
    print("   attention_plot = am.get_asset_path('transformers', 'results', 'attention_weights.png')")
    print("   embedding_analysis = am.get_asset_path('transformers', 'results', 'embedding_analysis.png')")
    print("   # Generate plots...")
    print("   am.register_asset('transformers', 'results', 'attention_weights.png', attention_plot)")
    print("   am.register_asset('transformers', 'results', 'embedding_analysis.png', embedding_analysis)")
    print("   ```")

if __name__ == "__main__":
    demo_asset_management()
    demo_project_usage()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, List
import os
import sys

# Add the parent directory to the path to import asset_manager
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from asset_manager import AssetManager

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SimpleCNN(nn.Module):
    """Simple CNN that processes images with local convolutions"""
    
    def __init__(self, input_channels=3, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers - process local regions
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layers - reduce spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Store intermediate activations for visualization
        activations = []
        
        # Conv1: Process local 3x3 regions
        x = F.relu(self.conv1(x))
        activations.append(x.clone())
        x = self.pool(x)
        
        # Conv2: Process larger local patterns
        x = F.relu(self.conv2(x))
        activations.append(x.clone())
        x = self.pool(x)
        
        # Conv3: Process even larger patterns
        x = F.relu(self.conv3(x))
        activations.append(x.clone())
        x = self.pool(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, activations

class SimpleVisionTransformer(nn.Module):
    """Simple Vision Transformer that processes images with global attention"""
    
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=64, depth=3, heads=8):
        super(SimpleVisionTransformer, self).__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        
        # Patch embedding
        self.patch_embedding = nn.Linear(self.patch_dim, dim)
        
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(depth)
        ])
        
        # Classification head
        self.mlp_head = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Create patches
        patches = self.create_patches(x)  # (batch_size, num_patches, patch_dim)
        
        # Embed patches
        x = self.patch_embedding(patches)  # (batch_size, num_patches, dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, dim)
        
        # Add position embedding
        x = x + self.pos_embedding
        
        # Store attention maps for visualization
        attention_maps = []
        
        # Process through transformer blocks
        for transformer in self.transformer_blocks:
            x, attention = transformer(x)
            attention_maps.append(attention)
        
        # Use class token for classification
        x = x[:, 0]  # Take class token
        x = self.mlp_head(x)
        
        return x, attention_maps
    
    def create_patches(self, x):
        """Convert image to patches"""
        batch_size, channels, height, width = x.shape
        patches = []
        
        for h in range(0, height, self.patch_size):
            for w in range(0, width, self.patch_size):
                patch = x[:, :, h:h+self.patch_size, w:w+self.patch_size]
                patch = patch.reshape(batch_size, -1)  # Flatten patch
                patches.append(patch)
        
        return torch.stack(patches, dim=1)

class TransformerBlock(nn.Module):
    """Single transformer block with self-attention"""
    
    def __init__(self, dim, heads):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        # Self-attention
        attn_out, attention = self.attention(x)
        x = self.norm1(x + attn_out)
        
        # MLP
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x, attention

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, dim, heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.proj(out)
        
        return out, attention

def create_sample_image(size=32):
    """Create a simple test image with distinct patterns"""
    # Create a simple image with different regions
    image = np.zeros((size, size, 3))
    
    # Top-left: Red square
    image[:size//4, :size//4, 0] = 1.0
    
    # Top-right: Green square
    image[:size//4, 3*size//4:, 1] = 1.0
    
    # Bottom-left: Blue square
    image[3*size//4:, :size//4, 2] = 1.0
    
    # Bottom-right: Yellow (red + green)
    image[3*size//4:, 3*size//4:, 0] = 1.0
    image[3*size//4:, 3*size//4:, 1] = 1.0
    
    # Center: White cross
    center_start = size//4
    center_end = 3*size//4
    image[center_start:center_end, center_start:center_end, :] = 0.5
    
    return image

def visualize_activations(activations, title):
    """Visualize CNN activations"""
    fig, axes = plt.subplots(1, len(activations), figsize=(15, 5))
    if len(activations) == 1:
        axes = [axes]
    
    for i, activation in enumerate(activations):
        # Take the first channel of the first sample
        act = activation[0, 0].detach().numpy()
        axes[i].imshow(act, cmap='viridis')
        axes[i].set_title(f'Layer {i+1} Activation')
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def visualize_attention(attention_maps, title):
    """Visualize Transformer attention maps"""
    fig, axes = plt.subplots(1, len(attention_maps), figsize=(15, 5))
    if len(attention_maps) == 1:
        axes = [axes]
    
    for i, attention in enumerate(attention_maps):
        # Take the first head of the first sample, class token attention
        attn = attention[0, 0, 0, 1:].detach().numpy()  # Skip class token
        # Reshape to image grid
        grid_size = int(np.sqrt(len(attn)))
        attn_grid = attn.reshape(grid_size, grid_size)
        axes[i].imshow(attn_grid, cmap='hot')
        axes[i].set_title(f'Layer {i+1} Attention')
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def demonstrate_differences():
    """Main demonstration function"""
    print("CNN vs Transformer Image Processing Demonstration")
    print("=" * 60)
    
    # Initialize asset manager
    am = AssetManager()
    
    # Create sample image
    sample_image = create_sample_image(32)
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(sample_image).unsqueeze(0).float()  # Add batch dimension and ensure float32
    
    print(f"Input image shape: {image_tensor.shape}")
    
    # Initialize models
    cnn_model = SimpleCNN()
    transformer_model = SimpleVisionTransformer()
    
    # Set models to evaluation mode
    cnn_model.eval()
    transformer_model.eval()
    
    print("\nProcessing image with CNN...")
    with torch.no_grad():
        cnn_output, cnn_activations = cnn_model(image_tensor)
    
    print("Processing image with Transformer...")
    with torch.no_grad():
        transformer_output, transformer_attention = transformer_model(image_tensor)
    
    # Visualize results
    plt.figure(figsize=(20, 10))
    
    # Original image
    plt.subplot(2, 4, 1)
    plt.imshow(sample_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # CNN activations
    for i, activation in enumerate(cnn_activations):
        plt.subplot(2, 4, i + 2)
        act = activation[0, 0].detach().numpy()
        plt.imshow(act, cmap='viridis')
        plt.title(f'CNN Layer {i+1}')
        plt.axis('off')
    
    # Transformer attention
    for i, attention in enumerate(transformer_attention):
        plt.subplot(2, 4, i + 5)
        attn = attention[0, 0, 0, 1:].detach().numpy()
        grid_size = int(np.sqrt(len(attn)))
        attn_grid = attn.reshape(grid_size, grid_size)
        plt.imshow(attn_grid, cmap='hot')
        plt.title(f'Transformer Layer {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save plot using asset manager
    plot_path = am.get_asset_path('neural_networks', 'plots', 'cnn_vs_transformer_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Register the plot asset
    am.register_asset('neural_networks', 'plots', 'cnn_vs_transformer_comparison.png', 
                     plot_path, {'type': 'comparison', 'models': ['CNN', 'Transformer']})
    print(f"📊 Plot saved and registered: {plot_path}")
    
    # Print key differences
    print("\n" + "="*60)
    print("KEY DIFFERENCES DEMONSTRATED:")
    print("="*60)
    print("\n1. CNN Processing:")
    print("   - Processes local regions with convolutions")
    print("   - Activations show local feature detection")
    print("   - Builds up from local to global patterns")
    print("   - Each layer focuses on increasingly larger local regions")
    
    print("\n2. Transformer Processing:")
    print("   - Processes entire image as patches")
    print("   - Attention maps show global relationships")
    print("   - Can attend to any part of the image from any other part")
    print("   - No assumption about spatial locality")
    
    print("\n3. Visual Evidence:")
    print("   - CNN activations: Local, structured patterns")
    print("   - Transformer attention: Global, distributed attention")
    print("   - CNN: Hierarchical feature extraction")
    print("   - Transformer: Parallel relationship modeling")
    
    # Show specific analysis
    print("\n" + "="*60)
    print("SPECIFIC ANALYSIS:")
    print("="*60)
    
    # Analyze CNN activations
    print("\nCNN Analysis:")
    for i, activation in enumerate(cnn_activations):
        act = activation[0, 0].detach().numpy()
        print(f"  Layer {i+1}: Max activation = {act.max():.3f}, "
              f"Mean activation = {act.mean():.3f}")
    
    # Analyze Transformer attention
    print("\nTransformer Analysis:")
    for i, attention in enumerate(transformer_attention):
        attn = attention[0, 0, 0, 1:].detach().numpy()
        print(f"  Layer {i+1}: Max attention = {attn.max():.3f}, "
              f"Mean attention = {attn.mean():.3f}")
        print(f"    Attention spread: {attn.std():.3f}")

if __name__ == "__main__":
    demonstrate_differences()

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm
from pathlib import Path

from src.data.dataset import get_dataloaders
from src.models.patchtst import PatchTST
from src.utils.config import cfg

def generate_tsne_plot():
    # 1. Setup
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data (Test Set)
    # Correcting call: get_dataloaders() returns (loaders_dict, all_codes)
    loaders, _ = get_dataloaders(batch_size=cfg.BATCH_SIZE)
    test_loader = loaders["test"]

    # 3. Load Model
    model = PatchTST().to(device)
    model_path = cfg.MODEL_DIR / "best_model.pt"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 4. Extract Latent Features
    all_features = []
    all_labels = []

    print("Extracting features from test set...")
    with torch.no_grad():
        for signals, _, super_labels, _ in tqdm(test_loader, desc="Processing"):
            signals = signals.to(device)
            # PatchTST forward returns (B, 12, d_model) features from encoder
            features, _ = model.encoder(signals) # (B, 12, d_model)
            latent = features.mean(dim=1).cpu().numpy() # (B, d_model)

            for i in range(len(super_labels)):
                active_classes = np.where(super_labels[i] == 1)[0]
                if len(active_classes) > 0:
                    dominant_idx = active_classes[0]
                    all_labels.append(cfg.SUPERCLASSES[dominant_idx])
                    all_features.append(latent[i])

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    # 5. t-SNE reduction
    print(f"Running t-SNE on {len(all_features)} samples...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(all_features)

    # 6. Plotting (IEEE style)
    plt.figure(figsize=(10, 8))
    sns.set_style("white")
    palette = sns.color_palette("bright", len(cfg.SUPERCLASSES))

    scatter = sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=all_labels,
        palette=palette,
        alpha=0.6,
        edgecolor="w",
        s=50
    )

    plt.title("Fig. 6: t-SNE Visualization of PatchTST Latent Features", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(title="ECG Superclasses", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    output_path = cfg.PLOT_DIR / "latent_tsne.png"
    cfg.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Success! t-SNE plot saved to: {output_path}")

if __name__ == "__main__":
    generate_tsne_plot()

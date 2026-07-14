"""
Plot training and validation losses from TensorBoard events for comparison.
"""
import glob
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def main():
    log_dir = '/home/jyuan/.stable-wm/savi_cjepa_pusht/tb_logs'
    log_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not log_files:
        print("No event files found in", log_dir)
        return

    print("Reading events from:", log_files[0])
    ea = event_accumulator.EventAccumulator(log_files[0])
    ea.Reload()

    # Available tags check
    tags = ea.scalars.Keys()
    print("Available scalar tags:", tags)

    # Extract Epoch level values
    # Total loss
    train_loss = [s.value for s in ea.Scalars('epoch/loss')] if 'epoch/loss' in tags else []
    val_loss = [s.value for s in ea.Scalars('val/loss')] if 'val/loss' in tags else []
    
    # Reconstruction loss
    train_recon = [s.value for s in ea.Scalars('epoch/recon_loss')] if 'epoch/recon_loss' in tags else []
    val_recon = [s.value for s in ea.Scalars('val/recon_loss')] if 'val/recon_loss' in tags else []

    # Ignore the first training epoch for better scale detail
    train_loss = train_loss[1:]
    val_loss = val_loss[1:]
    train_recon = train_recon[1:]
    val_recon = val_recon[1:]

    epochs = list(range(2, len(train_loss) + 2))

    # Setup matplotlib figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Total Loss
    if train_loss:
        ax1.plot(epochs, train_loss, label='Train Total Loss', marker='o', linewidth=2, color='#1f77b4')
    if val_loss:
        # Match length of validation loss with epochs
        ax1.plot(epochs[:len(val_loss)], val_loss, label='Val Total Loss', marker='s', linewidth=2, color='#ff7f0e')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.set_title('Total Loss (Reconstruction + KL)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Reconstruction Loss
    if train_recon:
        ax2.plot(epochs, train_recon, label='Train Recon Loss (MSE)', marker='o', linewidth=2, color='#2ca02c')
    if val_recon:
        ax2.plot(epochs[:len(val_recon)], val_recon, label='Val Recon Loss (MSE)', marker='s', linewidth=2, color='#d62728')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss Value', fontsize=12)
    ax2.set_title('Reconstruction Loss (MSE)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle('StoSAVi PushT Training Progress Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    out_path = '/home/jyuan/.gemini/antigravity-ide/brain/84ca7224-7271-41ba-9541-8c03fd85f0e9/loss_comparison.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print("Successfully saved plot to:", out_path)

if __name__ == '__main__':
    main()

import glob
import json
from tensorboard.backend.event_processing import event_accumulator

def main():
    log_files = glob.glob('/home/jyuan/.stable-wm/savi_cjepa_pusht/tb_logs/events.out.tfevents.*')
    ea = event_accumulator.EventAccumulator(log_files[0])
    ea.Reload()

    train_recon = [s.value for s in ea.Scalars('epoch/recon_loss')]
    val_recon = [s.value for s in ea.Scalars('val/recon_loss')]
    train_total = [s.value for s in ea.Scalars('epoch/loss')]
    val_total = [s.value for s in ea.Scalars('val/loss')]

    res = []
    for i in range(len(train_total)):
        vt = val_total[i] if i < len(val_total) else None
        vr = val_recon[i] if i < len(val_recon) else None
        res.append({
            'epoch': i + 1,
            'train_total': train_total[i],
            'val_total': vt,
            'train_recon': train_recon[i],
            'val_recon': vr
        })

    with open('/home/jyuan/.gemini/antigravity-ide/brain/84ca7224-7271-41ba-9541-8c03fd85f0e9/losses.json', 'w') as f:
        json.dump(res, f, indent=2)
    print("Saved losses to json!")

if __name__ == '__main__':
    main()

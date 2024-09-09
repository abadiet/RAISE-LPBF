from dataset.dataset import SamplesScanLines
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torch.nn import MSELoss
import datetime
from math import sqrt


def collate_fn(batch):
    return torch.stack([x[0] for x in batch]), torch.stack([x[1][0:2] for x in batch])


def evaluate(
    model,
    loader,
    device,
    power_mean=215,
    power_std=80,
    speed_mean=900,
    speed_std=400
):
    criterion = MSELoss()
    total_loss = 0
    total_mse_speed = 0
    total_mse_power = 0
    total_nmse_speed = 0
    total_nmse_power = 0

    with torch.no_grad():
        batch_i = 1
        for frames, labels in loader:
            if batch_i % 100 == 0:
                print(f"{datetime.datetime.now()}: Batch {batch_i}/{len(loader)}")
            batch_i += 1

            frames = frames.to(device)
            labels = labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            outputs[:, 0] = outputs[:, 0] * speed_std + speed_mean
            outputs[:, 1] = outputs[:, 1] * power_std + power_mean
            labels[:, 0] = labels[:, 0] * speed_std + speed_mean
            labels[:, 1] = labels[:, 1] * power_std + power_mean

            error = outputs - labels
            error_norm = error / labels.abs()

            nmse_speed = error_norm[:,0].pow(2).mean()
            nmse_power = error_norm[:,1].pow(2).mean()
            mse_speed = error[:,0].pow(2).mean()
            mse_power = error[:,1].pow(2).mean()
            total_nmse_speed += nmse_speed.item()
            total_nmse_power += nmse_power.item()
            total_mse_speed += mse_speed.item()
            total_mse_power += mse_power.item()

    print(f'Loss: {total_loss / len(loader)}')
    print(f'NMSE Speed: {total_nmse_speed / len(loader)}')
    print(f'NMSE Power: {total_nmse_power / len(loader)}')
    print(f'MSE Speed: {total_mse_speed / len(loader)}')
    print(f'MSE Power: {total_mse_power / len(loader)}')
    print(f'RMSE Speed: {sqrt(total_mse_speed / len(loader))}')
    print(f'RMSE Power: {sqrt(total_mse_power / len(loader))}')


if __name__ == '__main__':

    validation_sz = 100000

    batch_size = 32
    Nframes = 32
    model_id = 888
    i_epoch = 11
    model_path = f'/home/tabadie/workstation/RAISE-LPBF/model/CNN3D/save/v1/{Nframes}_model{model_id}/epochs/{i_epoch}/model.pth'
    dataset_path = f'/home/tabadie/workstation/RAISE-LPBF/model/CNN3D/save/v1/val_dataset.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = list(range(torch.cuda.device_count()))
    num_workers = 10

    model = torch.load(model_path, weights_only=False)
    model.to(device)
    model.eval()

    dataset = SamplesScanLines(Nframes=Nframes, version=1.0)
    val_indices = torch.load(dataset_path, weights_only=False)
    val_dataset = Subset(dataset, val_indices)
    print(f"Dataset size: {len(val_dataset)}")
    if validation_sz is not None:
        val_dataset, _ = random_split(val_dataset, [validation_sz, len(val_dataset) - validation_sz])
        print(f"Reduced dataset size: {len(val_dataset)}")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, prefetch_factor=2, collate_fn=collate_fn)

    evaluate(model, val_loader, device)

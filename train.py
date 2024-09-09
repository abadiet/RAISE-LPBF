from model import CNN3D
from dataset.dataset import SamplesScanLines
import torch
from torch.utils.data import DataLoader, random_split, Subset
from matplotlib import pyplot as plt
from torch.nn import MSELoss
from torch.nn.parallel import DataParallel
import os
import numpy as np


def collate_fn(batch):
    return torch.stack([x[0] for x in batch]), torch.stack([x[1][0:2] for x in batch])


if __name__ == '__main__':

    Nframes = 32
    height = 41
    width = 41
    batch_size = 32

    log_path = f"save/v1/{Nframes}_model63/"
    if not os.path.exists(log_path + "epochs/"):
        os.makedirs(log_path + "epochs/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = list(range(torch.cuda.device_count()))
    num_workers = 32

    load_path = "/home/tabadie/workstation/RAISE-LPBF/model/CNN3D/save/v1/" # None if no set to load
    validation_sz = 100000

    # Load data
    print("Dataset loading...")
    dataset = SamplesScanLines(Nframes=Nframes, height=height, width=width, version=1.0)
    print(f"Dataset size: {len(dataset)}")
    print("Loaders loading...")
    if load_path is None:
        train_size = int(0.7 * len(dataset))
        print(" > splitting...")
        train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        torch.save(train_dataset.indices, log_path + "train_dataset.pth")
        torch.save(val_dataset.indices, log_path + "val_dataset.pth")
    else:
        print(" > loading from {}".format(load_path))
        train_dataset = Subset(dataset, torch.load(load_path + "train_dataset.pth"))
        val_dataset = Subset(dataset, torch.load(load_path + "val_dataset.pth"))

    if validation_sz is not None:
        val_dataset, _ = random_split(val_dataset, [validation_sz, len(val_dataset) - validation_sz])
        print(f"Reduced validation set size: {len(val_dataset)}")

    print(" > training loader...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, prefetch_factor=2, collate_fn=collate_fn)
    print(" > validating loader...")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, prefetch_factor=2, collate_fn=collate_fn)
    print("Data loaded.")

    # Setup model
    num_epochs = 100
    model = CNN3D(Nframes, height, width)
    model = DataParallel(model.to(device), device_ids=device_ids)
    criterion = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Loop
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs): 

        model.train()
        running_loss = 0
        for frames, labels in train_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(frames)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss.append(running_loss / len(train_loader))

        model.eval()
        running_loss = 0
        for frames, labels in val_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            out = model(frames)
            loss = criterion(out, labels)

            running_loss += loss.item()

        val_loss.append(running_loss / len(train_loader))

        print(f'Epoch {epoch+1}\tTraining loss: {train_loss[-1]}\tValidation loss: {val_loss[-1]}')

        if not os.path.exists(log_path + f"epochs/{epoch + 1}"):
            os.makedirs(log_path + f"epochs/{epoch + 1}")
        torch.save(model, log_path + f"epochs/{epoch + 1}/model.pth")
        torch.save(model.state_dict(), log_path + f"epochs/{epoch + 1}/model_weights.pth")
        torch.save(optimizer, log_path + f"epochs/{epoch + 1}/opt.pth")

        np.save(log_path + "training_loss.npy", train_loss)
        np.save(log_path + "validating_loss.npy", val_loss)

        plt.plot(range(1,len(train_loss)+1), train_loss, label='Training Loss')
        plt.plot(range(1,len(val_loss)+1), val_loss, label='Validation Loss')
        plt.legend()
        plt.xlabel("Number of epochs")
        plt.ylabel("Training Loss")
        plt.savefig(log_path + "training_loss.png")
        plt.clf()

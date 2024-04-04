import numpy as np
import matplotlib.pyplot as plt
import math
from fsspec import open as fsopen
import h5py
from torch.utils.data import Dataset
import torchvision.transforms as tvt
import torch
from torch.nn import Sequential, ReLU, Linear, Module, Identity


class FramesSP(Dataset):
    def __init__(self, data_fp, test=False, repeat_frames=True, xoffset=20, yoffset=20, power_mean=215, speed_mean=900, **_):
        self.repeat_frames = repeat_frames
        self.xoffset, self.yoffset = xoffset, yoffset
        self.speed_mean, self.power_mean = speed_mean, power_mean
        self.data_fp = data_fp
        self.test = test

        with h5py.File(self.data_fp, 'r') as h5f:
            self.layer_cumsums = [np.cumsum(    # nb of lines for every object
                [len(np.unique(h5f[o][l]['scan_line_index'][:])) for l in h5f[o].keys()]  # nb of lines for every object o's layers
            ) for o in h5f.keys()]
        self.obj_cumsum = np.cumsum([o[-1] for o in self.layer_cumsums])    # total nb of lines
        self._len = self.obj_cumsum[-1]

    def __getitem__(self, index):
        # retrieve object, layer, scanline
        object_i = np.argwhere(self.obj_cumsum - index - 1 >= 0)[0][0]
        rindex = index if object_i == 0 else index - self.obj_cumsum[object_i - 1]
        layer_i = np.argwhere(self.layer_cumsums[object_i] - 1 - rindex >= 0)[0][0]
        rindex = rindex if layer_i == 0 else rindex - self.layer_cumsums[object_i][layer_i - 1]
        scan_line_i = rindex
        with h5py.File(self.data_fp, 'r') as h5f:
            object = list(h5f.keys())[object_i]
            layer = list(h5f[object].keys())[layer_i]
            indices = np.where(h5f[object][layer]['scan_line_index'][:] == scan_line_i)
            frames = h5f[object][layer]['frame'][indices]
            if not self.test:
                speed, power = h5f[object][layer]['laser_params'][scan_line_i]

        # crop frames around max intensity of mean frame
        i, j = np.unravel_index(frames.mean(0).argmax(), frames[0].shape)
        x = tvt.functional.crop(torch.tensor(np.array([frames])), i - self.xoffset, j - self.yoffset, 2 * self.xoffset + 1, 2 * self.yoffset + 1)
        if self.repeat_frames:
            x = x.repeat_interleave(3, dim=0)

        if self.test:
            y = torch.tensor([None, None, object_i, layer_i, scan_line_i])
        else:
            y = torch.tensor([speed / self.speed_mean, power / self.power_mean, object_i, layer_i, scan_line_i])

        return x, y

    def __len__(self):
        return self._len


class CNN3DResnet(Module):
    def __init__(self, fc_type='original', retrain_depth=0, fc_depth=5, weights_fp=None, strict_weights=True, **kwargs):
        super(CNN3DResnet, self).__init__()
        self.fc_depth = fc_depth

        self.cnn = torch.hub.load('facebookresearch/pytorchvideo:main', 'slow_r50', pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = False
        if retrain_depth > 0:
            for param in self.cnn.blocks[-retrain_depth:].parameters():
                param.requires_grad = True
        elif retrain_depth == -1:
            for param in self.cnn.blocks.parameters():
                param.requires_grad = True

        if fc_type == 'single':
            self.cnn.blocks[-1].proj = Linear(2048, 1)
            self.linear_layers = Identity()
        elif fc_type == 'original':
            self.linear_layers = Sequential(
                Linear(400, 200),
                ReLU(inplace=True),
                *(l for _ in range(self.fc_depth) for l in (Linear(200, 200), ReLU(inplace=True))),
                Linear(200, 2),
            )

        if weights_fp is not None:
            self.load_weights(weights_fp, strict=strict_weights)

    def forward(self, x):
        x = self.cnn(x)
        x = self.linear_layers(x)
        return x

    def load_weights(self, fp, strict=True):
        self.load_state_dict(torch.load(fp), strict=strict)


def show_frames_of_line(frames, label, cols = 0, rows = 0):
    N_frames = len(frames)
    if (cols * rows > 1):
        frame_idx_step = math.floor(N_frames / (cols * rows - 1))
        if frame_idx_step == 0: frame_idx_step = 1
    elif (cols * rows < 1):
        cols = math.floor(math.sqrt(N_frames))
        rows = math.ceil(N_frames / cols)
        frame_idx_step = 1
    cur_i = 0
    figure = plt.figure(figsize=(cols, rows))
    while (cur_i < cols * rows and (frame_idx := cur_i * frame_idx_step) < N_frames):
        figure.add_subplot(rows, cols, cur_i + 1)
        plt.axis("off")
        plt.title("%i / %i"%(frame_idx + 1, N_frames))
        plt.imshow(frames[frame_idx].squeeze())
        cur_i += 1
    figure.suptitle("object={:n} layer={:n} line={:n} (speed_reduced={}, power_reduced={})".format(label[2], label[3], label[4], label[0], label[1]))
    plt.show()


if __name__ == '__main__':
    dataset = FramesSP("RAISE_LPBF_train_C027_layer0202.hdf5", repeat_frames=False)
    print ("{0} lines loaded".format(len(dataset)))

    line_idx = torch.randint (len(dataset), size=(1,)).item()
    frames, label = dataset[line_idx]
    frames = frames[0]
    show_frames_of_line(frames, label)

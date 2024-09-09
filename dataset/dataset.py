import numpy as np
import h5py
from torch.utils.data import Dataset
import torchvision.transforms as tvt
import torch
import os
import datetime


class ScanLines(Dataset):
    """Handle the raw video"""

    def __init__(self, load_path="/home/SharedFiles/tabadie/RAISE-LPBF/dataset/", height=41, width=41, test=False, crop=True, version=1.1):
        self.half_height = height // 2
        self.height = height
        self.half_width = width // 2
        self.width = width
        self.test = test
        self.load_path = load_path
        self.crop = crop
        self.version = version

        if version == 1.0:
            self.h5f = h5py.File(os.path.join(load_path, 'v1.0/RAISE_LPBF_test.hdf5' if test else 'v1.0/RAISE_LPBF_train.hdf5'), 'r')
            self.h5f_keys = self.h5f.keys()
        elif version == 1.1:
            # might be better to use h5repack
            if not test:
                indices = [27, 28, 30, 31, 33, 34, 36, 37, 38, 39, 41, 42]
            else:
                indices = [29, 32, 35, 40]
            self.h5f = [h5py.File(os.path.join(load_path, 'v1.1/C0{}.hdf5'.format(i)), 'r') for i in indices]
            self.h5f_keys = range(len(self.h5f))
        else:
            raise ValueError("Version not supported")

        # retrieving length
        self.layer_cumsums = [np.cumsum(    # nb of lines for every object
            [len(np.unique(self.h5f[o][l]['scan_line_index'][:])) for l in self.h5f[o].keys()]  # nb of lines for every object o's layers
        ) for o in self.h5f_keys]
        self.obj_cumsum = np.cumsum([o[-1] for o in self.layer_cumsums])
        self._len = self.obj_cumsum[-1]

    def __getitem__(self, index):
        if index < 0:
            index += self._len
        # retrieve object, layer, scanline
        object_i = np.argwhere(self.obj_cumsum - index - 1 >= 0)[0][0]
        rindex = index if object_i == 0 else index - self.obj_cumsum[object_i - 1]
        layer_i = np.argwhere(self.layer_cumsums[object_i] - 1 - rindex >= 0)[0][0]
        scan_line_i = rindex if layer_i == 0 else rindex - self.layer_cumsums[object_i][layer_i - 1]
        object = list(self.h5f_keys)[object_i]
        layer = list(self.h5f[object].keys())[layer_i]
        indices = np.where(self.h5f[object][layer]['scan_line_index'][:] == scan_line_i)
        frames = self.h5f[object][layer]['frame'][indices]
        frames = torch.tensor(np.array(frames), dtype=torch.float)
        if not self.test:
            speed, power = self.h5f[object][layer]['laser_params'][scan_line_i]

        # crop frames around max intensity of mean frame
        if self.crop:
            i, j = torch.unravel_index(frames.mean(dim=0).argmax(), frames[0].shape)
            frames = tvt.functional.crop(
                frames,
                i - self.half_height,
                j - self.half_width,
                self.height,
                self.width
            )

        if self.test:
            labels = torch.tensor([torch.nan, torch.nan, object_i, layer_i, scan_line_i], dtype=torch.float)
        else:
            labels = torch.tensor([speed, power, object_i, layer_i, scan_line_i], dtype=torch.float)

        return frames, labels

    def __len__(self):
        return self._len


class CleanScanLines(ScanLines):
    """Handle the clean video"""

    def __init__(
            self,
            frame_max_thresh=100,
            frames_mean=0.45,
            frames_std=0.225,
            power_mean=215,
            power_std=80,
            speed_mean=900,
            speed_std=400,
            **kwargs
        ):
        super().__init__(**kwargs)

        self.frame_max_thresh = frame_max_thresh
        self.power_mean = power_mean
        self.power_std = power_std
        self.speed_mean = speed_mean
        self.speed_std = speed_std

        self.process = tvt.Compose([
            tvt.Lambda(lambda x: x / 255.0),
            tvt.Normalize([frames_mean], [frames_std])
        ])

        index_fp = os.path.join(self.load_path, f'v{self.version}/clean_sl_index_map_{"test" if self.test else "train"}_{frame_max_thresh}.npy')
        if not os.path.isfile(index_fp):
            print(f"Computing index map {index_fp}.\nThis may take a while.")
            self.__create_index_map(index_fp)

        self.__index_map = np.load(index_fp)
        self._len = self.__index_map[-1, 1]

    def __create_index_map(self, index_fp):
        print(f"Computing index map {index_fp}.\nThis may take a while.\nStarting at {datetime.datetime.now()}")
        first_index = 0
        frames, _ = super().__getitem__(first_index)
        is_clean = (frames.reshape(-1, self.height * self.width).max(dim=1)[0] > self.frame_max_thresh)
        n = torch.diff(is_clean).sum()
        while n == 0 and not is_clean[0] and first_index + 1 < len(self):
            first_index += 1
            frames, _ = super().__getitem__(first_index)
            is_clean = (frames.reshape(-1, self.height * self.width).max(dim=1)[0] > self.frame_max_thresh)
            n = torch.diff(is_clean).sum()

        if (first_index >= len(self)):
            raise ValueError("No clean frames found")

        index_map = []
        index_map.append([first_index, 0])

        for index in range(first_index, len(self)):
            frames, _ = super().__getitem__(index)
            is_clean = (frames.reshape(-1, self.height * self.width).max(dim=1)[0] > self.frame_max_thresh)
            n = torch.diff(is_clean).sum()
            if n == 0 and not is_clean[0]:
                # There is no clean frame
                index_map.append([index, index_map[-1][1] + index - index_map[-1][0] - 1])
            elif n > 1:
                first = is_clean[0]
                last = is_clean[-1]
                n += (first != last)
                n //= 2
                n += (first == last) * first
                # There is n groups of following clean frames
                if n > 1:
                    index_map.append([index, index_map[-1][1] + index - index_map[-1][0] - 1 + n])
            # if index_map has not been udpated, it means that the frames contain an unique group of following clean frames
            if index % 10000 == 0:
                print(f"{datetime.datetime.now()}: index {index}")

        index_map.append([len(self), index_map[-1][1] + len(self) - index_map[-1][0] - 1])
        index_map = np.array(index_map)
        np.save(index_fp, index_map)

    def __getitem__(self, index):
        if index < 0:
            index += self._len
        if index == 0:
            true_index = self.__index_map[0, 0]
            offset = 0
        else:
            imap = np.argwhere(self.__index_map[:,1] - index >= 0)[0][0]
            true_index_before, fake_index_before = self.__index_map[imap - 1]
            true_index_after = self.__index_map[imap, 0]
            true_index = true_index_before + index - fake_index_before
            offset = true_index - true_index_after
            if offset > 0:
                true_index = true_index_after
            else:
                offset = 0

        frames, labels = super().__getitem__(true_index)
        is_clean = (frames.reshape(-1, self.height * self.width).max(dim=1)[0] > self.frame_max_thresh)
        is_clean = torch.concat((torch.tensor([is_clean[0]]), is_clean))
        groups = torch.cumsum(torch.diff(is_clean), dim=0)
        igroup = 2 * offset
        igroup += (is_clean[0] == False)
        frames = frames[groups == igroup]

        if (len(frames) == 0):
            raise IndexError(f"Not enough group of frames for index {index}")

        frames = self.process(frames)
        if not self.test:
            labels[0] = (labels[0] - self.speed_mean) / self.speed_std
            labels[1] = (labels[1] - self.power_mean) / self.power_std

        return frames, labels


class SamplesScanLines(CleanScanLines):
    """Handle constant lenght clean video"""

    def __init__(self, Nframes, **kwargs):
        super().__init__(**kwargs)

        self.Nframes = Nframes

        index_fp = os.path.join(self.load_path, f'v{self.version}/samples_index_map_{"test" if self.test else "train"}_{self.frame_max_thresh}_{Nframes}.npy')
        if not os.path.isfile(index_fp):
            print(f"Computing index map {index_fp}.\nThis may take a while.\nStarting at {datetime.datetime.now()}")
            self.__create_index_map(index_fp)

        self.__index_map = np.load(index_fp)
        self._len = self.__index_map[-1, 1]

    def __create_index_map(self, index_fp):
        div_buf = [k // self.Nframes for k in range(0, 550)]

        first_index = 0
        frames, _ = super().__getitem__(first_index)
        n = div_buf[len(frames)]
        while n == 0 and first_index + 1 < len(self):
            first_index += 1
            frames, _ = super().__getitem__(first_index)
            if (len(frames) >= len(div_buf)):
                print(f"len(frames): {len(frames)}, len(div_buf): {len(div_buf)}")
                for i in range(len(div_buf), len(frames) + 1):
                    div_buf.append(i // self.Nframes)
            n = div_buf[len(frames)]

        if (first_index >= len(self)):
            raise ValueError("No clean frames found")

        index_map = []
        index_map.append([first_index, 0])

        for index in range(first_index, len(self)):
            frames, _ = super().__getitem__(index)
            if (len(frames) >= len(div_buf)):
                print(f"len(frames): {len(frames)}, len(div_buf): {len(div_buf)}")
                for i in range(len(div_buf), len(frames) + 1):
                    div_buf.append(i // self.Nframes)
            n = div_buf[len(frames)]
            if (n == 0):
                index_map.append([index, index_map[-1][1] + index - index_map[-1][0] - 1])
            elif (n > 1):
                index_map.append([index, index_map[-1][1] + index - index_map[-1][0] - 1 + n])
            if index % 10000 == 0:
                print(f"{datetime.datetime.now()}: index {index}")

        index_map.append([len(self), index_map[-1][1] + len(self) - index_map[-1][0] - 1])
        index_map = np.array(index_map)
        np.save(index_fp, index_map)

    def __getitem__(self, index):
        if index < 0:
            index += self._len
        if index == 0:
            true_index = self.__index_map[0, 0]
            offset = 0
        else:
            imap = np.argwhere(self.__index_map[:,1] - index >= 0)[0][0]
            true_index_before, fake_index_before = self.__index_map[imap - 1]
            true_index_after = self.__index_map[imap, 0]
            true_index = true_index_before + index - fake_index_before
            offset = true_index - true_index_after
            if offset > 0:
                true_index = true_index_after
            else:
                offset = 0

        frames, labels = super().__getitem__(true_index)
        frames = frames[offset * self.Nframes : (offset + 1) * self.Nframes]

        return frames, labels


class CleanFrames(CleanScanLines):
    """Handle clean frames"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        index_fp = os.path.join(self.load_path, f'v{self.version}/clean_frames_index_map_{"test" if self.test else "train"}_{self.frame_max_thresh}.npy')
        if not os.path.isfile(index_fp):
            print(f"Computing index map {index_fp}.\nThis may take a while.\nStarting at {datetime.datetime.now()}")
            self.__create_index_map(index_fp)

        self.__index_map = np.load(index_fp)
        self._len = self.__index_map[-1]

    def __create_index_map(self, index_fp):
        index_map = np.cumsum([len(super(CleanFrames, CleanFrames).__getitem__(self, index)[0]) for index in range(super(CleanFrames, CleanFrames).__len__(self))])
        np.save(index_fp, index_map)

    def __getitem__(self, index):
        if index < 0:
            index += self._len
        true_index = np.argwhere(self.__index_map - index - 1 >= 0)[0][0]
        frames, labels = super().__getitem__(true_index)
        offset = index if true_index == 0 else index - self.__index_map[true_index - 1]
        return frames[offset], labels

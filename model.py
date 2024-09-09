from torch.nn import Conv3d, Linear, Module, MaxPool3d
from torch.nn.functional import relu

class CNN3D(Module):
    def __init__(self, Nframes, height, width):
        super(CNN3D, self).__init__()
        self.conv1 = Conv3d(1, 64, kernel_size=5, padding='same')
        self.pool1 = MaxPool3d(kernel_size=3)
        self.fc1_in_sz = 64 * (Nframes//3) * (height//3) * (width//3)
        self.fc1 = Linear(self.fc1_in_sz, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = relu(self.conv1(x))
        x = self.pool1(x)
        x = x.view(-1, self.fc1_in_sz)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x

import torch.nn as nn

class LowerConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_blocks = nn.Sequential()  # Use nn.Sequential to hold blocks

        in_channels = 1  # Start with 1 input channel (grayscale)
        for out_channels, num_repeats in [(8, 4), (16, 3), (32, 3)]:
            for _ in range(num_repeats):
                self.conv_blocks.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
                )
                self.conv_blocks.append(nn.ReLU())
                self.conv_blocks.append(nn.BatchNorm2d(out_channels))
                in_channels = out_channels  # Update in_channels for next block

            self.conv_blocks.append(nn.MaxPool2d(2, 2))

    def forward(self, x):
        return self.conv_blocks(x)
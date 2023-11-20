""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=False, scale_channels=1):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = (DoubleConv(n_channels, 32//scale_channels, dropout=0.1))
        self.down1 = (Down(32//scale_channels, 64//scale_channels, dropout=0.1))
        self.down2 = (Down(64//scale_channels, 128//scale_channels, dropout=0.2))
        self.down3 = (Down(128//scale_channels, 256//scale_channels, dropout=0.3))
        factor = 2 if trilinear else 1
        self.up1 = (Up(256//scale_channels, (128//scale_channels) // factor, trilinear, dropout=0.1))
        self.up2 = (Up(128//scale_channels, (64//scale_channels) // factor, trilinear, dropout=0.2))
        self.up3 = (Up(64//scale_channels, (32//scale_channels) // factor, trilinear, dropout=0.3))
        self.outc = (OutConv(32//scale_channels, n_classes))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        def print_memory_usage():
            NO_PRINT = False
            if self.device.type == 'cuda' and not NO_PRINT:
                print(f'Current memory allocated: {torch.cuda.memory_allocated(self.device)/1024**3:.2f} GB')
                print(f'Max memory allocated: {torch.cuda.max_memory_allocated(self.device)/1024**3:.2f} GB')
                print(f'Current memory cached: {torch.cuda.memory_reserved(self.device)/1024**3:.2f} GB')
                print(f'Max memory cached: {torch.cuda.max_memory_reserved(self.device)/1024**3:.2f} GB')

        print_memory_usage()
        x1 = self.inc(x)
        print("Memory usage after inc function:")
        print_memory_usage()

        x2 = self.down1(x1)
        print("Memory usage after down1 function:")
        print_memory_usage()

        x3 = self.down2(x2)
        print("Memory usage after down2 function:")
        print_memory_usage()

        x4 = self.down3(x3)
        print("Memory usage after down3 function:")
        print_memory_usage()

        x = self.up1(x4, x3)
        print("Memory usage after up1 function:")
        print_memory_usage()

        x = self.up2(x, x2)
        print("Memory usage after up2 function:")
        print_memory_usage()

        x = self.up3(x, x1)
        print("Memory usage after up3 function:")
        print_memory_usage()

        logits = self.outc(x)
        print("Memory usage after outc function:")
        print_memory_usage()

        return logits

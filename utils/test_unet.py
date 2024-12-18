# Test for the UNet model
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from utils.unet import UNet


def test_unet():
    print("Starting test...")
    double_conv = UNet.double_conv(256, 256)
    print("Double Conv Layer:", double_conv)
    input_image = torch.rand((1, 3, 512, 512))  # Reduced input size
    print("Input Image Shape:", input_image.shape)
    model = UNet(3, 10)
    output = model(input_image)
    print("Output Size:", output.size())
    print("Test completed.")


if __name__ == "__main__":
    test_unet()

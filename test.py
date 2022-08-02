from tkinter import TOP
from bs4 import TemplateString
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import save_image
from einops.layers.torch import Rearrange

temp_img = Image.open('datasets/avenue/testing/01/0000.jpg')
temp_img.save('test/origin.jpg',"JPEG")
tt = ToTensor()
tp = ToPILImage()
tensor_img = tt(temp_img)
print(tensor_img.shape)

tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)
print(tensor_img.shape)

rot_img = tensor_img.rot90(1, dims=[3,4]).reshape([3,640,360])
print(rot_img.shape)

rot_pil = tp(rot_img)

rot_pil.save('test/rotation.jpg',"JPEG")

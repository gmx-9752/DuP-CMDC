import torch
import numpy as np


def np_to_torch_img(img_np):
    img_pt = torch.tensor(img_np.astype("float") / 255)
    img_pt = img_pt.permute(2, 0, 1).unsqueeze(0).half().to("cuda")
    return img_pt


def pil_to_torch_img(img_pil):
    return np_to_torch_img(np.array(img_pil))


def torch_to_np_img(img):
    return img[0].permute(1, 2, 0).clip(0, 1).detach().cpu().numpy()


def np_to_pil_img(img):
    from PIL import Image

    return Image.fromarray((img * 255).astype("uint8"))


def torch_to_pil_img(img):
    return np_to_pil_img(torch_to_np_img(img))

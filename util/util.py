import numpy as np
import torch as th


def img2tensor(img):
    tensor = img.transpose(2, 0, 1)
    tensor = tensor[np.newaxis, :]

    return th.Tensor(tensor)


def tensor2img(tensor):
    img = tensor.numpy()
    img = np.squeeze(img, 0)
    img = img.transpose(1, 2, 0)

    return img

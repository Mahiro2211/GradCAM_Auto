import argparse
import cv2
import numpy as np
import torch
import timm
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
import os
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
import ipywidgets as widgets
import io
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from CrossFormer import CrossFormer
import warnings

warnings.filterwarnings("ignore")
import glob
from modeling import VisionTransformer, VIT_CONFIGS

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./image', help='the path of image')
parser.add_argument('--method', default='all', help='the method of GradCam can be specific ,default all')
parser.add_argument('--aug_smooth', default=True, choices=[True, False],
                    help='Apply test time augmentation to smooth the CAM')
parser.add_argument('--use_cuda', default=True, choices=[True, False],
                    help='if use GPU to compute')
parser.add_argument(
    '--eigen_smooth',
    default=False, choices=[True, False],
    help='Reduce noise by taking the first principle componenet'
         'of cam_weights*activations')
parser.add_argument('--modelname', default="ViT-B-16", help='Any name you want')


def grab_image(path: str):
    # 获取文件夹下的所有图片文件
    image_files = glob.glob(os.path.join(path, '*.jpg')) + glob.glob(os.path.join(path, '*.png'))
    for image_file in image_files:
        print(image_file)
    return image_files


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def ONE(path: str, cam, method: str, eigen_smooth=True, aug_smooth=True, index=1, ifall=False, modelname=None):
    rgb_img = cv2.imread(path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=None,
                        eigen_smooth=eigen_smooth,
                        aug_smooth=aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]
    if ifall:
        base_path = f'{modelname}all_img/{method}_save_img'
        os.makedirs(f'{modelname}all_img/{method}_save_img', exist_ok=True)
    else:
        base_path = f'{modelname}_{method}_save_img'
        os.makedirs(f'{modelname}_{method}_save_img', exist_ok=True)

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f'{base_path}/{index}_{method}_cam.jpg', cam_image)


def creat_cam(method, methods):
    if method == "ablationcam":
        cam = methods[method](model=model,
                              target_layers=target_layers,
                              use_cuda=t.use_cuda,
                              reshape_transform=reshape_transform,
                              ablation_layer=AblationLayerVit())
    else:
        cam = methods[method](model=model,
                              target_layers=target_layers,
                              use_cuda=t.use_cuda,
                              reshape_transform=reshape_transform)
    return cam


if __name__ == '__main__':

    t, _ = parser.parse_known_args()

    # -----Creat-Model----- (take ViT B16 as a example)
    vit_config = VIT_CONFIGS["ViT-B_16"]
    vit = VisionTransformer(vit_config, zero_head=False, hash_bit=64, vis=True)
    # weight = torch.load('ViT-B_16.npz')
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load("coco_HashNet_ViT-B_16_Bit64-BestModel.pt")
    # checkpoint['net'].pop('hash_layer.3.weight')
    # checkpoint['net'].pop('hash_layer.3.bias')
    vit.load_state_dict(checkpoint['net'], strict=False)
    model = vit
    model.eval()
    # ---------------------

    # -----Loading-Pretrained-Weight-----
    # weight = torch.load('ViT-B_16.npz')
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load("coco_HashNet_ViT-B_16_Bit64-BestModel.pt")
    # checkpoint['net'].pop('hash_layer.3.weight')
    # checkpoint['net'].pop('hash_layer.3.bias')
    vit.load_state_dict(checkpoint['net'], strict=False)
    model = vit
    model.eval()
    # -----------------------------------

    # -----Select-Target_Layer-----------
    target_layers = [model.transformer.encoder.layer[-1].attention_norm]
    # -----------------------------------
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         #  "fullgrad": FullGrad
         }
    # -----Feed-Into-GradCAM-------------
    image_file = grab_image(t.path)
    print(f'grab image_file shown as {image_file}')

    from tqdm import tqdm

    if t.method not in list(methods.keys()):
        for key, value in methods.items():
            # print(key,value)
            for i, path in enumerate(image_file):
                cam = creat_cam(key, methods)
                ONE(path=path, index=i, aug_smooth=t.aug_smooth, eigen_smooth=t.eigen_smooth, method=key, cam=cam,
                    ifall=True, modelname=t.modelname)
    else:
        cam = creat_cam(t.method, methods)
        for i, path in enumerate(image_file):
            ONE(path=path, index=i, aug_smooth=t.aug_smooth, eigen_smooth=t.eigen_smooth, method=t.method, cam=cam,
                modelname=t.modelname)

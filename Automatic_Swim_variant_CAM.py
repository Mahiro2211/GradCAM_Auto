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
from model.CrossFormer import CrossFormer
import warnings

warnings.filterwarnings("ignore")
import glob

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
parser.add_argument('--modelname', default="test", help='Any name you want')


def grab_image(path: str):
    # 获取文件夹下的所有图片文件
    image_files = glob.glob(os.path.join(path, '*.jpg')) + glob.glob(os.path.join(path, '*.png'))
    for image_file in image_files:
        print(image_file)
    return image_files


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
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

    # -----Creat-Model----- (take CrossFormer as an example)
    from model.crossFormer_args import get_config as get_configs
    from model.crossFormer_args import parse_option
    from model.CrossFormer import CrossFormer

    _, config = parse_option()
    model = CrossFormer(
        patch_size=config.MODEL.CROS.PATCH_SIZE,
        in_chans=config.MODEL.CROS.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=config.MODEL.CROS.EMBED_DIM,
        depths=config.MODEL.CROS.DEPTHS,
        num_heads=config.MODEL.CROS.NUM_HEADS,
        group_size=config.MODEL.CROS.GROUP_SIZE,
        mlp_ratio=config.MODEL.CROS.MLP_RATIO,
        qkv_bias=config.MODEL.CROS.QKV_BIAS,
        qk_scale=config.MODEL.CROS.QK_SCALE,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        ape=config.MODEL.CROS.APE,
        patch_norm=config.MODEL.CROS.PATCH_NORM,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        merge_size=config.MODEL.CROS.MERGE_SIZE,
        hash_bit=128)
    model.eval()
    # ---------------------

    # -----Loading-Pretrained-Weight-----
    checkpoint = torch.load("0.7665955556072438_128_model.pth", map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    # -----------------------------------

    # -----Select-Target_Layer-----------
    # target_layers = [model.layers[-1].blocks[-1].norm1,model.layers[-1].blocks[-1].norm2]
    # target_layers = [model.layers[-1].blocks[-1].norm2]
    target_layers = [model.layers[-1].blocks[-1].norm1]
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

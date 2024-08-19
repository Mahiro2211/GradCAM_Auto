# GradCAM_On_ViT
# 这是一个用于计算ViT及其变种模型的GradCAM自动脚本，可以自动处理批量的图像
A GradCAM automatic script to visualize the model result
# How to adjust your XXXFormer in GradCam
## Please ensure that your model is in a proper format.
<p>If the transformer you apply into is a swin'-like transformer(No Class Token) or ViT-like (Have a Class token)
 </p>
 <p>The shape of the tensor may look like <em>[Batch,49,768]</em> then you should deal with your model with the following steps to avoid some terrible <strong>RuntimeError</strong>
 </p>
 
```python

Class XXXFormer(nn.Moudle):
    def __init(self,...):
        super().__init__()
        .....
        self.avgpool = nn.AdaptiveAvgPool1d(1) #this is essential
    def forward(self,x):
        x = self.forward_feartrue(x) # Supose that the out put is [Batch,49,768]
        x = self.avgpool(x.transpose(1,2)) # [Batch,49,768] --> [Batch,768,49] --> [Batch,768,1]
        x = torch.flatten(x,1) # [Batch,768]
```
## Get Your Target Layer
<p>Find your last transformer block and select the LayerNorm() attribute as your target layer if you have more than one LayerNorm() attribute you can get them all in a list or just select one of them</p>
<p> Your target layer may look like</p>
 
 ```python
# choose one LayerNorm() attribute for your target layer
target_Layer1 = [vit.block[-1].norm1]
target_Layer2 = [vit.block[-1].norm2]
# or stack up them all
target_Layer3 = [vit.block[-1].norm1,vit.block.norm2]
 ```
### Why do we choose LayerNorm as the target layer? 
### Reference: On the Expressivity Role of LayerNorm in Transformer's Attention (ACL 2023).
<p>The reason may be like this as shown in the picture</p>

![image](https://github.com/Mahiro2211/GradCAM_Automation/assets/130811701/eba4b15e-bda6-4f2d-b4b0-8999385f787f)




* Automatic_Swim_variant_CAM.py
* Automatic_ViT_variant_CAM.py
 
the two .py file shown above is the main Python script you need to run
just set up your image file and run these two scripts!!

### Using EigenCam as an example

![Result](https://github.com/Mahiro2211/GradCAM_Automation/assets/130811701/4fb5c2df-da8c-4748-9a28-7bf39f3d8b1b)


### Param you need to Pay attention

```python
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
```

|Method|
|-----|
| CrossFormer (ICLR 2022) |
| Vision Transformer (ICLR 2021) |

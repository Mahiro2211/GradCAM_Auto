# GradCAM_On_ViT_Variant
A GradCAM automatic script to visualize the model result
# How to adjust your XXXFormer in GradCam
<p>If the transformer you apply into is a swin'-like transformer(No Class Token) or ViT-like (Have a Class token)
 </p>
 <p>The shape of the tensor may look like <em>[Batch,49,768]</em> then you should deal with your model with the following steps to avoid some terrible **RuntimeError**
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
* Automatic_Swim_variant_CAM.py
* Automatic_ViT_variant_CAM.py
 
the two .py file shown above is the main Python script you need to run
just set up your image file and run these two scripts!!
![Result](https://github.com/Mahiro2211/GradCAM_Automation/assets/130811701/4fb5c2df-da8c-4748-9a28-7bf39f3d8b1b)


|Method|
|-----|
| CrossFormer(ICLR 2022) |
| Vision Transformer (ICLR2021) |

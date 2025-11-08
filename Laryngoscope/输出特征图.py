import os
import torch
import cv2
import numpy as np
from PIL import Image
from torch import nn
import torchvision.models as tm
from torchvision import models
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.resnet = tm.resnet18()
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(1000,256)
        self.linear1 = nn.Linear(256,1)
        self.sigmoid = nn.Sigmoid()
        # Register hooks for each layer to capture the feature maps
        for name, layer in self.resnet.named_modules():
            if isinstance(layer, nn.Conv2d):  # Register hook only for Conv2d layers
                layer.register_forward_hook(self.save_feature_maps(name))

    def save_feature_maps(self, layer_name):
        def hook(module, input, output):
            self.feature_maps.append((layer_name, output))

        return hook

    def forward(self, img):
        feature= self.resnet(img)
        feature = self.linear(feature)
        feature = torch.nn.functional.normalize(feature,dim=1)
        cls = self.linear1(feature).squeeze()
        cls = torch.sigmoid(cls)
        return feature, cls

    def clear_feature_maps(self):
        self.feature_maps = []


def save_feature_maps_to_images(feature_maps, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for layer_name, feature_map in feature_maps:
        feature_map = feature_map[0].detach().cpu().numpy()  # Assuming batch size of 1 for visualization
        for i in range(feature_map.shape[0]):
            img = feature_map[i]
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
            img = (img * 255).astype(np.uint8)  # Scale to [0, 255]
            if img.ndim == 2:  # If single channel, convert to 3-channel grayscale
                img = np.stack([img]*3, axis=-1)
            img = Image.fromarray(img)
            img.save(os.path.join(output_dir, f"{layer_name}_feature_map_{i}.png"))


# Example usage
model = Net()
model.load_state_dict(torch.load('/Users/lixiaofan/Desktop/项目/原型不确定性困难气道评估/protoairway/Laryngoscope/Larycheck/300.pth', map_location='cpu'))  # Load your trained model

# Your input tensor here (no preprocessing)
img = cv2.imread('/Users/lixiaofan/Desktop/项目/原型不确定性困难气道评估/protoairway/original_data_whole/154/7.png', cv2.IMREAD_COLOR)
img = cv2.resize(img, (224, 224))
img = img.transpose(2, 0, 1) / 255
img = torch.from_numpy(img).float().unsqueeze(dim=0)

# Forward pass and save feature maps
model.clear_feature_maps()
model(img)
save_feature_maps_to_images(model.feature_maps, '/Users/lixiaofan/Desktop/OT困难气道论文和代码/OT困难气道/特征可视化/喉镜154/')

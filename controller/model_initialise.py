import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()

        # Convolutional layer
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 
                              256, 256, 256, 'M', 
                              512, 512, 512
                              ]
        
        # Deep learning layer
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = self.make_layers(self.frontend_feat)
        self.backend = self.make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        if load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            # Load VGG16 frontend weights
            self.frontend.load_state_dict(mod.features[0:23].state_dict())
        else:
            self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        layers = []
        d_rate = 4 if dilation else 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, 
                                   padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.GELU()]
                else:
                    layers += [conv2d, nn.Mish()]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CSRNet().to(device)
checkpoint = torch.load("CSRNet.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)  # add batch dim
    return img



def preprocess_image(path,l1,l2):
    # img = Image.open(path).convert("RGB")
    image = cv2.imread(path)
    image = cv2.resize(image, (1024, 576), interpolation=cv2.INTER_CUBIC)
    height,width,_ = image.shape
    top = (height//2)-l1
    bottom = (height//2)+l2
    img = image[top:bottom, 0:width]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0)  # add batch dimension


    return np.array(image), img_tensor,height,width


# ------------------------------
# 4. Inference &  Visualization
# ------------------------------
def run_inference(image_path,l1,l2):
    # Load and preprocess
    orig_img, img_tensor, height, width = preprocess_image(image_path,l1,l2)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)   # density map

        density_map = output.squeeze().cpu().numpy()
        density_map = np.clip(density_map, 0, None)
        print(density_map,'density map')
        count = np.sum(density_map)

    print(f"Estimated Crowd Count: {int(count)}")
    cv2.putText(orig_img, f"Crowd Count: {int(count)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Show with OpenCV
    cv2.line(orig_img, (0, (height//2) - l1), (width, (height//2) - l1),
                         (0, 255, 0), thickness=2)
    cv2.line(orig_img, (0, (height//2) + l2), (width, (height//2) + l2),
                         (0, 255, 0), thickness=2)
    cv2.imshow("Crowd Density Estimation", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



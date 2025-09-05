import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import threading
import time
import socketio
# Setting up socket for communication

sio = socketio.Client()

@sio.event
def connect():
    print("Connected to Node.js Socket.IO server")

sio.connect("http://localhost:5050")

def send_data(alert):
    sio.emit('alert', alert)
    print("Alert sent to Node.js server:", alert)



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
checkpoint = torch.load("controller/CSRNet.pth", map_location=device)
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



def preprocess_image(image,l1,l2):
    # img = Image.open(path).convert("RGB")
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
def run_inference(frame,l1,l2):
    # Load and preprocess
    orig_img, img_tensor, height, width = preprocess_image(frame,l1,l2)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)   # density map

        density_map = output.squeeze().cpu().numpy()
        density_map = np.clip(density_map, 0, None)
        count = np.sum(density_map)

    return count



# Store active camera streams
cameras = {}       # { cam_id: VideoCapture }
frames = {}        # { cam_id: latest frame (bytes) }
locks = {}         # { cam_id: threading.Lock() }
running = {}
threshold={}       # { threshold: bool }
camera_count={}
peoplec_count={}



def camera_loop(cam_id, stream_url,l1,l2,threshold_value=40):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Cannot open {stream_url}")
        return

    running[cam_id] = True
    threshold[cam_id] = threshold_value
    locks[cam_id] = threading.Lock()
    camera_count[cam_id] = 0
    peoplec_count[cam_id] = 0

    while running[cam_id]:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1024, 576), interpolation=cv2.INTER_CUBIC)
        if(camera_count[cam_id]%10==0):
            peoplec_count[cam_id] = run_inference(frame,l1,l2)
            check_count(cam_id)
            print(f"Estimated Crowd Count: {int(peoplec_count[cam_id])} for Camera ID: {cam_id}")
        
        camera_count[cam_id] += 1
        

        height, width, _ = frame.shape
        cv2.putText(frame, f"Estimated Crowd Count: {int(peoplec_count[cam_id])} for Camera ID: {cam_id}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        # Show with OpenCV
        cv2.line(frame, (0, (height//2) - l1), (width, (height//2) - l1),
                            (0, 255, 0), thickness=2)
        cv2.line(frame, (0, (height//2) + l2), (width, (height//2) + l2),
                            (0, 255, 0), thickness=2)
        _, buffer = cv2.imencode(".jpg", frame)
        with locks[cam_id]:
            frames[cam_id] = buffer.tobytes()
        time.sleep(0.03)
    

    cap.release()

    with locks[cam_id]:
        frames.pop(cam_id, None)


def generate_frames(cam_id):
    while True:
        if cam_id not in frames:
            continue
        with locks[cam_id]:
            frame = frames[cam_id]
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        

def stop_camera_fun(cam_id):
    if cam_id in running:
        running[cam_id] = False
        print(f"Stopping camera {cam_id}")
    else:
        print(f"Camera {cam_id} not found") 


def check_count(camera_id):
    print(f"Camera ID: {camera_id} has count {peoplec_count[camera_id]}")
    if peoplec_count[camera_id] > threshold[camera_id]:
        # Trigger alert
        print("########################## Threshold exceeded! Triggering alert, stampede possibility...")
        send_data({"alert": f"Threshold exceeded! Triggering alert, stampede possibility!! for camera {camera_id}.","type":"alert","severity":"critical","camera_id":camera_id})
    else:
        percent = (peoplec_count[camera_id]/threshold[camera_id])*100
        if percent>80:
            min_cam = min(peoplec_count, key=peoplec_count.get)
            print(f"Camera with minimum count: {min_cam} ({peoplec_count[min_cam]})")
            if(min_cam != camera_id):
                print("---------------------------------------")
                print(f"Consider redirecting some crowd to camera {min_cam} area.")
                # send_data({"alert": f"Consider redirecting some crowd to camera {min_cam} area.","type":"alert","camera_id":camera_id})
                print("---------------------------------------")


# send_data({"alert": f"Threshold exceeded! Triggering alert, stampede possibility!! for camera 1.","type":"alert","severity":"critical","camera_id":1})

"""
Example:
    python -m Data.human_segmentation
"""

import os

P365_PATH = 'Data/places365'

from PIL import Image
import torchvision
from torchvision import transforms as T

# urllib.error.URLError: <urlopen error [WinError 10060]
from six.moves import urllib
proxy = urllib.request.ProxyHandler({'http': '70.10.15.10:8080'})
proxy = urllib.request.ProxyHandler({'https': '70.10.15.10:8080'})
opener = urllib.request.build_opener(proxy)
urllib.request.install_opener(opener)

# Download the pretrained Faster R-CNN model from torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the class names given by PyTorch's official Docs
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Defining a function for get a prediction result from the model
def get_prediction(img_path, threshold):
    img = Image.open(img_path) # Load the image
    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    pred = model([img]) # Pass the image to the model
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())

    # IndexError: list index out of range
    pred_score_index = [pred_score.index(x) for x in pred_score if x > threshold]
    if len(pred_score_index) == 0:
        return

    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    
    for pred in pred_class:
        if pred == 'person':
            os.remove(img_path)
            return


if __name__ == '__main__':
    # Example: After detection
    dir_names = os.listdir(P365_PATH)
    for dir_name in dir_names:
        class_names = os.listdir(os.path.join(P365_PATH, dir_name))
        for class_name in class_names:
            im_names = os.listdir(os.path.join(P365_PATH, dir_name, class_name))
            print("Number of images of class %s: %d" % (class_name, len(im_names)))
            for im_name in im_names:
                print('Process image: %s/%s' % (class_name, im_name))
                get_prediction(os.path.join(P365_PATH, dir_name, class_name, im_name), threshold=0.8)

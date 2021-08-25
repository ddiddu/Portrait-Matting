"""
Run ONNX model of MODNet with

Example:
    Background replacement:
        python -m onnx.run_onnx --onnx-path=model/mnv2_vol1_SPS_matte_aug_all_val_mse.onnx --bgd=replace --ofd=False
        
    Background blur:
        python -m onnx.run_onnx --onnx-path=model/modnet_kd_stability_supervisely_all_100_20.onnx --bgd=blur --ofd=True
"""

import argparse
import cv2, time
import onnxruntime as rt
import numpy as np

rt.set_default_logger_severity(3)

# Width and height
video_width  = 640 
video_height = 360

# Image width and height
img_width   = 256
img_height  = 256

# Mask width and height
mask_width  = 256
mask_height = 256

# Frame rate
fps = ""
elapsedTime = 0

### average fps
fps_sum = 0
fps_num = 0
fps_avg = ""

# Load background image
bgd = cv2.resize(cv2.imread('samsung.png'), (video_width, video_height))
bgd = cv2.cvtColor(bgd, cv2.COLOR_BGR2RGB) / 255.0

# Video capturer
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)

parser = argparse.ArgumentParser()
parser.add_argument('--onnx-path', type=str, help='path of pre-trained MODNet')
parser.add_argument('--bgd', type=str, help='background replacement or background blur')
parser.add_argument('--ofd', type=str, help='use of one frame delay')
args = parser.parse_args()

# onnxruntime
sess = rt.InferenceSession(args.onnx_path)
sess_input = sess.get_inputs()[0]
sess_input_name = sess_input.name
sess_input_shape = sess_input.shape
print(f'{sess_input}') # NodeArg(name='input_1', type='tensor(float)', shape=['N', 320, 320, 3])

sess_output = sess.get_outputs()[0]
sess_output_name = sess_output.name
sess_output_shape = sess_output.shape
print(f'{sess_output}') # NodeArg(name='flatten_1', type='tensor(float)', shape=['N', 16320])

masks = []

while True :
    t1 = time.time()

    # Read frames
    ret, frame = cap.read()

    # Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    simg = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)
    simg = simg.reshape((1, img_height, img_width, 3)) / 255.0
    simg = simg.transpose(0, 3, 1, 2).astype('float32')

    outputs = sess.run(None, {sess_input_name: simg})[0]
    
    """ Display the mask when alpha > 0.5 """
    mask = np.float32((outputs > 0.9)).reshape((mask_height, mask_width, 1))

    """ Alpha blend frame with background """
    # mask = outputs.reshape((mask_height, mask_width, 1))

    """ OFD: One Frame Delay """
    if args.ofd == "True":
        masks.append(mask)
        if len(masks) == 3:
            prv = masks.pop(0)
            now = masks[0]
            adj = masks[1]
        
            for i in range (mask_height):
                for j in range (mask_width):
                    if now[i][j] != prv[i][j]: # O(n^2)
                        if prv[i][j] == adj[i][j]:
                            now[i][j] = prv[i][j]
        
            mask = now

    """ Post-process """
    mask = cv2.GaussianBlur(mask, (3, 3), 1)
    img = cv2.resize(img, (video_width, video_height)) / 255.0
    mask = cv2.resize(mask, (video_width, video_height)).reshape((video_height, video_width, 1))

    """ Background replacement """ 
    if args.bgd == 'replace':
        frame = (img * mask) + (bgd * (1 - mask))
    """ Background blur """
    if args.bgd == 'blur':
        frame = (img * mask) + (cv2.blur(img, (20, 20)) * (1 - mask))
    frame = np.uint8(frame * 255.0)
    
    elapsedTime = time.time() - t1

    # Print frame rate
    fps = "{:.1f} FPS".format(1/elapsedTime)
    cv2.putText(frame, fps, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), thickness=2)

    # Display the resulting frame
    cv2.imshow('portrait segmentation', frame[..., ::-1])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    """ Average FPS """
    fps_num+=1

    if not fps_num==1:
        fps_sum+=1/elapsedTime
        fps_avg="{:.1f} FPS".format(fps_sum/(fps_num-1))
    
    print("fps average = ", str(fps_avg))


cap.release()
cv2.destroyAllWindows()

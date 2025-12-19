import torch
import torchvision.transforms as transforms
import cv2
from collections import deque
import common_utils as utils
import time

prev_frame_time = 0
cur_frame_time = 0
frame_time_deque = deque(maxlen=30) 
# model = utils.BaseModel()
# model.load_state_dict(torch.load("base_model/checkpoint_6.pth", map_location=utils.device))
# model.to(utils.device)
model = torch.jit.load("pruned_model/jit_quantized.pth")
# model.compile()
transform = transforms.Normalize((0.5), (0.5))

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.resize(grayscale, (48, 48))
    grayscale = cv2.resize(grayscale, (96, 96))
    grayscale = torch.from_numpy(grayscale).float()
    grayscale = torch.unsqueeze(torch.unsqueeze(grayscale, 0), 0)
    grayscale = transform(grayscale)

    compute_start = time.time()
    output = model(grayscale)
    compute_end = time.time()
    fps = 1 / (compute_end - compute_start)
    _, predicted = torch.max(output.data, 1)
    label = utils.label_to_text[predicted[0].item()]

    frame_time_deque.append(fps)
    fps_text = f"FPS: {int(sum(frame_time_deque) / len(frame_time_deque))}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    label_text = f"Label: {label}"
    cv2.putText(frame, label_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(5)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")

import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch

from src import model
from src import util
from src.body import Body
from src.hand import Hand

# 0️⃣ 选用 GPU（若存在）
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on", device)           # 可留作调试

# 1️⃣ 加载模型到 GPU

body_estimation = Body("model/body_pose_model.pth")
hand_estimation = Hand("model/hand_pose_model.pth")

body_estimation.model = body_estimation.model.to(device)
hand_estimation.model = hand_estimation.model.to(device)

# 2️⃣ 若有你自己转成 Tensor 的代码，也记得 .to(device)
#    OpenPose 官方实现内部会自动迁移，不需要改。

# print(f"Torch device: {torch.cuda.get_device_name()}")

if torch.cuda.is_available():
    print("Torch device:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("CUDA 不可用，改用 CPU")
    device = torch.device("cpu")
# model.to(device) / torch.load(..., map_location=device)




cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
while True:
    ret, oriImg = cap.read()
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        all_hand_peaks.append(peaks)

    canvas = util.draw_handpose(canvas, all_hand_peaks)

    cv2.imshow('demo', canvas)#一个窗口用以显示原视频
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


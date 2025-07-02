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

test_image = 'images/20250627BOY.jpg'
oriImg = cv2.imread(test_image)  # B,G,R order
candidate, subset = body_estimation(oriImg)

canvas = copy.deepcopy(oriImg)
canvas = util.draw_bodypose(canvas, candidate, subset)
# detect hand
hands_list = util.handDetect(candidate, subset, oriImg)

all_hand_peaks = []
for x, y, w, is_left in hands_list:
    # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # if is_left:
        # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
        # plt.show()
    peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
    peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
    peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
    # else:
    #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
    #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
    #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
    #     print(peaks)
    all_hand_peaks.append(peaks)

canvas = util.draw_handpose(canvas, all_hand_peaks)

# Matplotlib：plt.savefig()用来保存图片
import matplotlib.pyplot as plt
import time, os
out_dir = "snapshots"
os.makedirs(out_dir, exist_ok=True)
fname = os.path.join(out_dir, f"snap_{int(time.time())}.png")

plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.savefig(fname, bbox_inches="tight", pad_inches=0)   # ← 保存
plt.show()


# 保存最后一帧的节点数据
# candidate：所有检测到的关键点 (x, y, score)，形状 (N,3)
# subset：每个人用哪几个 candidate 索引组成，形状 (num_people,20)，前 18 列是对应关键点在 candidate 里的行号（找不到时是 -1）。
import scipy.io as sio
# 假设你在循环里，每帧都有 candidate, subset
# 这里只示例如何保存“最后一帧”
sio.savemat("pose_last_frame.mat", {
    "candidate": candidate,    # Nx3 double
    "subset":   subset         # Mx20 double
})
print("Saved pose_last_frame.mat")

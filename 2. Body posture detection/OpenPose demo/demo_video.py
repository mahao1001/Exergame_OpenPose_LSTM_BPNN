import copy
import json
import os
import subprocess
import argparse
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import ffmpeg  # pip install -U ffmpeg-python

# ───────────── OpenPose 估计器 ──────────────────────────────────────
from src import util
from src.body import Body
from src.hand import Hand

import torch

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


# ───────────── ffprobe 小工具 ─────────────────────────────────────
class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str


def ffprobe(path: str) -> FFProbeResult:
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",  # 避免 Windows GBK 解码错误
        )
        return FFProbeResult(res.returncode, res.stdout, res.stderr)
    except FileNotFoundError:
        return FFProbeResult(-1, "", "ffprobe executable not found")


def parse_avg_fps(rate_str: str) -> float:
    """
    '30000/1001' → 29.97 ; '30/1' → 30
    """
    try:
        num, den = map(int, rate_str.split("/"))
        return num / den if den else float(num)
    except Exception:
        return 30.0


# ───────────── 姿态处理 ────────────────────────────────────────────
def process_frame(frame, body=True, hands=True):
    canvas = frame.copy()
    if body:
        candidate, subset = body_estimation(frame)
        canvas = util.draw_bodypose(canvas, candidate, subset)

    if hands:
        hands_list = util.handDetect(candidate, subset, frame)
        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(frame[y : y + w, x : x + w])
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
            all_hand_peaks.append(peaks)

        canvas = util.draw_handpose(canvas, all_hand_peaks)
    return canvas


# ───────────── CLI 参数 ───────────────────────────────────────────
parser = argparse.ArgumentParser(description="Annotate poses in a video.")
parser.add_argument(
    "--file",
    type=str,
    default=r"images\20250618boxingMOD.mp4",
    help="Video file to process (default: images\\20250618boxingMOD.mp4)",
)
parser.add_argument("--no_hands", action="store_true", default=True, help="Skip hand pose")  #  default=True 就是关闭了手势检测
parser.add_argument("--no_body", action="store_true", help="Skip body pose")
args = parser.parse_args()

video_path = Path(args.file)
if not video_path.is_file():
    parser.error(f"Video not found: {video_path}")

cap = cv2.VideoCapture(str(video_path))

# ───────────── 读取输入视频信息 ────────────────────────────────────
probe = ffprobe(str(video_path))
if probe.return_code == 0 and probe.json:
    meta = json.loads(probe.json)
    vinfo = next(s for s in meta["streams"] if s["codec_type"] == "video")
    in_fps = parse_avg_fps(vinfo["avg_frame_rate"])
    in_pix_fmt = vinfo["pix_fmt"]
    in_codec = vinfo["codec_name"]
else:
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    in_pix_fmt, in_codec = "bgr24", "h264"

# 输出文件名
out_path = video_path.with_name(video_path.stem + ".processed" + video_path.suffix)

# ───────────── VideoWriter（基于 ffmpeg-python） ──────────────────
class Writer:
    def __init__(self, outfile: Path, fps: float, size, pix_fmt: str, vcodec: str):
        h, w = size
        outfile = str(outfile)

        # ❶ 若 w/h 为奇数 → 扩展到偶数
        if w % 2 == 1:
            w += 1
        if h % 2 == 1:
            h += 1
        self.out_width, self.out_height = w, h

        if Path(outfile).exists():
            os.remove(outfile)

        self.proc = (
            ffmpeg
            .input(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                s=f"{w}x{h}",
                r=str(fps),
            )
            .output(outfile, pix_fmt=pix_fmt, vcodec=vcodec)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def write(self, frame: np.ndarray):
        # ❷ 如果补齐过尺寸，用 cv2.copyMakeBorder 填充黑边
        fh, fw = frame.shape[:2]
        pad_right = self.out_width  - fw
        pad_bottom = self.out_height - fh
        if pad_right or pad_bottom:
            frame = cv2.copyMakeBorder(
                frame,
                top=0,
                bottom=pad_bottom,
                left=0,
                right=pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()



writer = None

# ───────────── 主循环 ───────────────────────────────────────────────
all_poses = []
frame_id  = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # —— 调试打印 ——
    print(f"[DEBUG] Processing frame {frame_id}")
    if not args.no_body:
        candidate, subset = body_estimation(frame)
        print(f"[DEBUG]  → {len(candidate)} keypoints, {subset.shape[0]} people")
    else:
        candidate, subset = None, None
        print("[DEBUG]  → body skipped")

    # —— 累积数据 ——
    all_poses.append({
        "frame":     frame_id,
        "candidate": candidate.tolist() if candidate is not None else [],
        "subset":    subset.tolist()    if subset    is not None else []
    })
    frame_id += 1

    # —— 原有可视化 + 写视频 ——
    posed = process_frame(
        frame, body=not args.no_body, hands=not args.no_hands
    )

    if writer is None:
        writer = Writer(out_path, in_fps, posed.shape[:2], in_pix_fmt, in_codec)

    cv2.imshow("Pose", posed)
    writer.write(posed)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if writer:
    writer.close()
cv2.destroyAllWindows()

# —— 写出 all_poses 到磁盘 ——
print("[DEBUG] Total frames accumulated:", len(all_poses))
out_json = video_path.with_name(video_path.stem + ".poses.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(all_poses, f, ensure_ascii=False, indent=2)
print("Saved all poses to", out_json)
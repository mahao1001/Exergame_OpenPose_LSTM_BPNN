import json
from pathlib import Path

# 1. 指定文件名
json_path = Path("images//20250618boxing.poses.json")
if not json_path.exists():
    print("❌ 文件不存在：", json_path.resolve())
    exit(1)

# 2. 加载 JSON
with open(json_path, "r", encoding="utf-8") as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        print("❌ JSON 解码失败：", e)
        exit(1)

# 3. 检查类型和长度
print("类型：", type(data))
if isinstance(data, list):
    print("总帧数（list 长度）：", len(data))
else:
    print("⚠️ JSON 最外层不是 list，可能是 dict：有", len(data), "个键")

# 4. 如果是 list，看看前几条内容
if isinstance(data, list) and data:
    print("\n--- 第 1 帧 ---")
    print(data[0])
    print("\n--- 第 2 帧 ---")
    print(data[1] if len(data)>1 else "<无第二帧>")
    print("\n字段 keys：", list(data[0].keys()))

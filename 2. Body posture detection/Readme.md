对 OpenPose姿态识别得到的数据处理
1. Python中运行得到的数据保存在了.json中；
2. 代码 data_pose_confidence.m 加载了.json数据，然后绘制了执行度曲线图
   （1）依次绘制单个曲线，整体的置信度和三个关键点的置信度曲线

3. 代码 posture_accuracy_score.m 对比了标准姿态和模仿者的姿态，计算了动作评分
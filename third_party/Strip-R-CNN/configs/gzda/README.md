# Strip R-CNN-S / gzda_v3_human_del

- 输入：原数据集的场景级 train/val/test 划分和 `instances_*_rbox.json`，类别为 building，旋转框为 le90。
- 训练：gpu5 的 CUDA 6，单进程、batch size 2、约 12 轮（90288 步）；脚本 `train_gpu7.sh` 使用已有 p2pformer conda 环境，并清理外部 PYTHONPATH。预训练骨干来自仓库 README 的官方 StripNet-S ImageNet 权重，保存在 `/data1/wzh26/model_zoo/stripnet_s_pretrain.pth.tar`。
- 空图：训练数据集和旋转增强均允许空标注。每 7524 步在验证集按旋转 IoU=0.5 计算 mAP，每 1000 步保存，保留最近 2 个常规 checkpoint 和 1 个验证集 mAP 最优 checkpoint。
- 去重：推理使用旋转 NMS，初始 IoU 阈值为 0.3；模型评估时保留低分候选（score_thr=0.05），不能把它当成已校准的部署阈值。模型训练完成后，应该仅用验证集比较置信度及 NMS 阈值，统计 precision、recall、误检及同一标注被多次检出的次数，再锁定阈值并在测试集做一次最终评估。
- 输出和日志：`work_dirs/gzda_strip_s/`。训练进度：`https://train.wzhpro.me` 中的 Strip R-CNN-S 条目。
- 恢复：在 gpu5 确认 CUDA 6 没有其他用户后运行 `bash configs/gzda/train_gpu5.sh`；脚本会从最新 checkpoint 自动恢复。

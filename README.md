# YOLOv11n-P2 (VisDrone) 微小目标检测

基于 YOLOv11n 的无人机航拍微小目标检测，加入 P2 高分辨率检测头与空洞卷积以提升微小目标召回；支持消融实验、评估与 SAHI 切片推理。

## 亮点
- P2 高分辨率检测头 + 可选空洞卷积（Dilated）增强上下文感受野。
- 一键消融：Baseline / P2 / P2+Dilated 自动训练与对比。
- 推理双路径：原生 YOLO + SAHI 切片。
- 复现性：入口脚本默认设定随机种子。

## 目录结构
```
CV/Project/
├── VisDrone.yaml              # 数据集配置（含自动下载脚本）
├── yolo11n.yaml               # Baseline 配置
├── yolov11n-p2.yaml           # P2 检测头配置
├── yolov11n-p2-dilated.yaml   # P2 + 空洞卷积配置
├── ablation_study.py          # 消融实验主脚本（推荐）
├── start_train.py             # 单模型训练
├── eval.py                    # 评估脚本（动态读取类别名）
├── demo_inference.py          # 推理对比（SAHI）
├── convert_visdrone_to_yolo.py# 标注转换与清洗
├── technical_details.md       # 技术细节
└── README.md                  # 本文件
```

## 环境与依赖
- 安装 Python 3.8+，GPU 训练推荐安装 CUDA 对应的 PyTorch。
- 安装依赖：
```powershell
pip install ultralytics sahi opencv-python pillow tqdm numpy
```

## 数据集介绍
本项目使用 **VisDrone-DET2019** 数据集，这是无人机航拍目标检测领域的权威数据集，专门用于应对“图大物小”和“背景复杂”的检测挑战。

- **数据来源**：[VisDrone-DET2019](http://aiskyeye.com/)
- **数据规模**：共包含 10,209 张静态图像（6,471 张训练集 / 548 张验证集 / 3,190 张测试集）。
- **场景特点**：
  - **高分辨率**：图像尺寸通常较大（如 2000×1500），包含大量微小目标（< 32×32 像素）。
  - **复杂环境**：涵盖不同天气、光照条件下，以及拥挤的城市道路、广场等场景。
- **目标类别** (10类)：
  `pedestrian` (行人), `people` (人), `bicycle` (自行车), `car` (汽车), `van` (杂货车), `truck` (卡车), `tricycle` (三轮车), `awning-tricycle` (遮阳三轮车), `bus` (公交车), `motor` (摩托车)。

> **注意**：本项目提供的 `convert_visdrone_to_yolo.py` 转换脚本在生成标签时，会自动过滤掉 `Ignored regions` (0) 和 `Others` (11) 类别，并剔除严重遮挡 (`occlusion >= 2`) 或截断 (`truncation > 0.7`) 的低质量样本，以优化微小目标的训练效果。

## 数据准备和清洗
- 默认由 Ultralytics 在首次训练/评估时自动下载到 `datasets/VisDrone/`。
- 如需手动下载，请保持结构：`datasets/VisDrone/VisDrone2019-DET-{train,val,test-dev}/images`。
- 手动下载地址：
1.官网下载：[VisDrone](http://aiskyeye.com/)（需要注册账号）
2.[夸克网盘](https://pan.quark.cn/s/73ba175248ef)
- 若需要重新生成 YOLO 标签或清洗标注，运行：
```powershell
python convert_visdrone_to_yolo.py
```
  - 会裁剪越界框并过滤严重遮挡/截断的样本。

## 训练流程
### 推荐：一键消融实验
```powershell
# 逐个训练
python ablation_study.py train 1   # Baseline (yolo11n.yaml)
python ablation_study.py train 2   # P2 (yolov11n-p2.yaml)
python ablation_study.py train 3   # P2+Dilated (yolov11n-p2-dilated.yaml)

# 对比已完成实验
python ablation_study.py compare

# 一键跑3个模型并对比
python ablation_study.py train all


```
输出：`runs/ablation/*/weights/best.pt` 与 `runs/ablation/results_summary.json`。

### 备用：单模型快速训练（一般不用）
```powershell
python start_train.py   # 默认 yolov11n-p2-dilated.yaml，输出 runs/detect/visdrone_yolov11n_p2_dilated/
```
> 显存不足时，可在脚本内下调 `BATCH` 或 `IMGSZ`。

## 评估
```powershell
python eval.py
```
- 若未指定 `--model`，脚本会尝试读取 `runs/ablation/results_summary.json` 自动选择 mAP 最高的模型。
- 输出 mAP@0.5、mAP@0.5:0.95、AP_Small（若提供）、各类别 AP、推理速度估算，并生成可视化与 `predictions.json`。
- 评估结果路径：`runs\detect`。

## 推理
```powershell

# SAHI 切片推理（YOLOv11 兼容性有限，失败会自动跳过，只进行原生YOLO推理）
python demo_inference.py
```
输出：`demo_result/demo[推理次数]_[模型名]`。

## 复现性
- 入口脚本默认设置随机种子 42（`torch`/`numpy`/`random`）。

## 输出与重要路径
- 训练权重：`runs/ablation/*/weights/best.pt` 或 `runs/detect/visdrone_yolov11n_p2_dilated/weights/best.pt`
- 训练曲线：`runs/ablation/*/results.png` 或 `runs/detect/.../results.png`
- 消融汇总：`runs/ablation/results_summary.json`
- 推理结果：`demo_result/`
- 数据配置：`VisDrone.yaml`；模型配置：`yolo11n.yaml`、`yolov11n-p2.yaml`、`yolov11n-p2-dilated.yaml`

## 7. 常见问题
- **找不到数据/配置**：确认在项目根目录运行；确保 `VisDrone.yaml`、模型配置和数据集目录存在。
- **显存不足 (OOM)**：在训练脚本中下调 `batch`，或将 `imgsz` 改小；必要时关闭 `mosaic`。
- **SAHI 报错**：可能因 YOLOv11 适配不全，脚本会自动降级为原生推理；若必须切片，可自行调整 SAHI 版本或实现滑窗。
- **指标缺失**：确保 `results.csv` 生成；若仅有权重，先运行 `model.val()` 或重新训练生成指标。
- **复现性**：脚本默认设定随机种子 42；如需更强确定性，可启用 `torch.backends.cudnn.deterministic = True`（可能降速）。

## 致谢与引用
- YOLOv11: https://docs.ultralytics.com/
- SAHI: https://github.com/obss/sahi
- VisDrone: http://aiskyeye.com/

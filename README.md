# YOLOv11n-P2 (VisDrone) 微小目标检测

基于 YOLOv11n 的无人机航拍微小目标检测，加入 P2 高分辨率检测头与空洞卷积以提升微小目标召回；支持消融实验、评估与（可选）SAHI 切片推理。

## 亮点
- P2 高分辨率检测头 + 可选空洞卷积（Dilated）增强上下文感受野。
- 一键消融：Baseline / P2 / P2+Dilated 自动训练与对比。
- 推理双路径：原生 YOLO（默认）+ SAHI 切片（若兼容）。
- 复现性：入口脚本默认设定随机种子。

## 目录结构
```
CV/Project/
├── VisDrone.yaml              # 数据集配置（含自动下载脚本）
├── yolo11n.yaml               # Baseline 配置
├── yolov11n-p2.yaml           # P2 检测头配置
├── yolov11n-p2-dilated.yaml   # P2 + 空洞卷积配置
├── ablation_study.py          # 消融实验主脚本（推荐）
├── start_train.py             # 单模型训练（默认 P2+Dilated）
├── eval.py                    # 评估脚本（动态读取类别名）
├── demo_inference.py          # 推理对比（SAHI 兼容性降级）
├── convert_visdrone_to_yolo.py# 标注转换与清洗
├── technical_details.md       # 技术细节
└── README.md                  # 本文件
```

## 环境与依赖
```powershell
pip install ultralytics sahi opencv-python pillow tqdm numpy
```
> SAHI 对 YOLOv11 支持有限；脚本会在不兼容时自动跳过 SAHI，继续原生 YOLO 推理。
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

## 数据准备
- 默认由 Ultralytics 在首次训练/评估时自动下载并解压至 `datasets/VisDrone/`。
- 若手动下载，保持目录：`datasets/VisDrone/VisDrone2019-DET-{train,val,test-dev}/images`。
- 标注转换（若需重新生成 YOLO 标签）：
```powershell
python convert_visdrone_to_yolo.py
```
  - 会将框裁剪到图像范围，过滤严重遮挡/截断样本。

## 训练
### 消融实验（推荐）
```powershell
# 逐个训练
python ablation_study.py train 1   # Baseline
python ablation_study.py train 2   # P2
python ablation_study.py train 3   # P2+Dilated

# 或一键全跑
python ablation_study.py train all

# 对比已完成实验
python ablation_study.py compare
```
输出：`runs/ablation/*/weights/best.pt` 与 `runs/ablation/results_summary.json`。

### 单模型快速训练（备用）
```powershell
python start_train.py   # 默认 yolov11n-p2-dilated.yaml
```
> 显存不足可在脚本中下调 `batch`。

## 评估
```powershell
python eval.py --model runs/ablation/3_yolov11n_p2_dilated/weights/best.pt --split val
```
- 类别名从模型 `names` 动态读取，避免与数据集不一致。
- 输出 mAP@0.5、mAP@0.5:0.95、AP_small（若提供）、FPS 估算及各类 AP。

## 推理
```powershell
# 原生 YOLO（默认执行）
python demo_inference.py --model runs/ablation/3_yolov11n_p2_dilated/weights/best.pt \
    --image datasets/VisDrone/VisDrone2019-DET-test-dev/images/0000006_00159_d_0000005.jpg

# SAHI 切片（若兼容则执行，否则自动跳过）
python demo_inference.py --slice-height 640 --slice-width 640 --overlap 0.2 --conf 0.25
```
输出：`demo_result/sahi_result.jpg`（若 SAHI 成功）与 `demo_result/native_yolo/`。

## 复现性
- 入口脚本默认设置随机种子 42（`torch`/`numpy`/`random`）。
- 如需更强确定性，可手动启用 `torch.backends.cudnn.deterministic = True`（可能降速）。

## 常见问题
- **找不到配置/数据**：确保在 `CV/Project` 目录执行命令。
- **显存不足 (OOM)**：在脚本中下调 `batch`，或减小 `imgsz`。
- **SAHI 报错**：可能因 YOLOv11 兼容性，已自动降级为原生推理；若必须使用切片，可自行实现滑窗推理。
- **标注异常**：转换脚本已裁剪越界框并过滤重遮挡样本，必要时重新运行转换。

## 结果与消融预期（示例）
| 模型 | 检测头 | 空洞卷积 | 预期 mAP@0.5 |
|------|--------|---------|--------------|
| Baseline | P3/P4/P5 | ✗ | ~40-45% |
| P2 | P2/P3/P4/P5 | ✗ | ~48-52% |
| P2+Dilated | P2/P3/P4/P5 | ✓ | ~50-54% |

## 致谢与引用
- YOLOv11: https://docs.ultralytics.com/
- SAHI: https://github.com/obss/sahi
- VisDrone: http://aiskyeye.com/

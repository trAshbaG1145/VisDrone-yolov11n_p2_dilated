"""
模型评估脚本 - 评估训练好的模型在验证集/测试集上的性能

【作用】
- 评估训练好的模型在 VisDrone 数据集上的性能
- 计算详细的检测指标和类别级别的 AP
- 生成评估曲线和混淆矩阵

【主要功能】
1. 计算核心指标：mAP@0.5、mAP@0.5:0.95、Precision、Recall、FPS
2. 输出各类别 AP：VisDrone 10 类目标的详细性能
3. 生成可视化：PR 曲线、混淆矩阵、预测示例
4. 支持 CLI 参数：灵活指定模型、数据集、设备等

【使用场景】
- 评估单个模型的详细性能
- 收集论文实验数据
- 分析各类别的检测效果
- 对比不同模型在各类别上的表现

【用法】
  # 评估最佳模型
  python eval.py
  
  # 评估其他模型
  python eval.py --model runs/ablation/1_baseline_yolov11n/weights/best.pt
  
  # 在测试集上评估
  python eval.py --model <model_path> --split test --device 0

【输出内容】
  mAP@0.5     : 0.5234
  mAP@0.5:0.95: 0.3456
  Precision   : 0.6789
  Recall      : 0.5432
  各类别 AP@0.5: pedestrian, people, bicycle, car, ...

【特点】
- ✅ 支持 CLI 参数（灵活配置）
- ✅ 详细的类别级别指标
- ✅ 可用于论文实验数据收集
- ✅ 自动生成可视化结果
"""
import argparse
import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO  # type: ignore


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11n-P2 on VisDrone")
    # 修改：默认值设为 None，以便在 main 中判断是否需要自动查找
    parser.add_argument(
        "--model",
        default=None,
        help="Path to trained weights (leave empty to auto-select best model)",
    )
    parser.add_argument(
        "--data",
        default="VisDrone.yaml",
        help="Dataset YAML path",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size")
    parser.add_argument("--batch", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold")
    parser.add_argument("--device", default="0", help="Device id, e.g., '0' or 'cpu'")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate")
    return parser.parse_args()

def find_best_model_from_summary(project_dir="runs/ablation"):
    """
    读取消融实验汇总文件，寻找 mAP 最高的模型
    """
    summary_path = Path(project_dir) / "results_summary.json"
    
    if not summary_path.exists():
        return None, "未找到汇总文件 (results_summary.json)"

    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if not results:
            return None, "汇总文件为空"

        # 寻找 mAP (mAP@0.5:0.95) 最高的实验
        best_exp_name = None
        best_map = -1.0
        
        for exp_name, metrics in results.items():
            # 优先看 mAP@0.5:0.95，如果没有则看 mAP@0.5
            current_map = metrics.get('map', 0)
            if current_map > best_map:
                best_map = current_map
                best_exp_name = exp_name
        
        if best_exp_name:
            # 构建权重路径
            best_model_path = Path(project_dir) / best_exp_name / "weights" / "best.pt"
            if best_model_path.exists():
                return str(best_model_path), f"根据 mAP ({best_map:.4f}) 选中最佳模型: {best_exp_name}"
            else:
                return None, f"最佳模型权重文件丢失: {best_model_path}"
        
        return None, "无法从汇总中解析出最佳模型"

    except Exception as e:
        return None, f"读取汇总文件出错: {e}"

def main():
    args = parse_args()
    set_seed()
    
    # --- 模型路径选择逻辑 ---
    target_model = args.model
    
    # 1. 如果用户没有指定模型，尝试自动寻找最佳模型
    if target_model is None:
        print("🔍 用户未指定模型，正在寻找最佳模型...")
        best_model, msg = find_best_model_from_summary()
        
        if best_model:
            print(f"✅ {msg}")
            target_model = best_model
        else:
            print(f"⚠️ 自动选择失败: {msg}")
            # 2. 如果自动选择失败，回退到默认的 P2+Dilated 路径
            default_fallback = "runs/ablation/3_yolov11n_p2_dilated/weights/best.pt"
            print(f"🔄 回退使用默认路径: {default_fallback}")
            target_model = default_fallback

    # 3. 最终检查文件是否存在
    if not os.path.exists(target_model):
        print(f"❌ 错误: 模型文件不存在: {target_model}")
        print("💡 请先运行训练脚本: python ablation_study.py train all")
        return

    print("=" * 60)
    print(f"📊 模型评估 - {target_model} on VisDrone")
    print("=" * 60)

    # 加载模型
    print(f"\n📦 加载模型: {target_model}")
    model = YOLO(target_model)

    # 在验证集上评估
    print("\n🔍 开始评估...")
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,
        save_json=True,
    )

    # 输出关键指标
    print("\n" + "=" * 60)
    print("📈 评估结果")
    print("=" * 60)
    print(f"mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision   : {metrics.box.mp:.4f}")
    print(f"Recall      : {metrics.box.mr:.4f}")

    # 尝试获取尺度分布指标
    ap_small = getattr(metrics.box, "map_small", None)
    if ap_small is not None:
        print(f"AP_Small    : {ap_small:.4f}  (核心指标: <32x32像素)")

    # 计算推理速度 (FPS)
    if hasattr(metrics, "speed") and "inference" in metrics.speed:
        infer_ms = metrics.speed["inference"]
        fps = 1000.0 / infer_ms if infer_ms > 0 else 0
        print(f"FPS (估算)  : {fps:.2f}  (推理耗时 {infer_ms:.2f} ms)")

    print("-" * 60)
    
    # 按类别输出
    print("\n📊 各类别 AP@0.5:")
    print("-" * 60)
    class_names = getattr(model, "names", None) or {}
    for idx, ap in enumerate(metrics.box.ap50):
        name = class_names.get(idx, f"class_{idx}") if isinstance(class_names, dict) else str(idx)
        print(f"{idx:2d}. {name:20s}: {ap:.4f}")

    print("\n" + "=" * 60)
    print("✅ 评估完成!")


if __name__ == "__main__":
    main()

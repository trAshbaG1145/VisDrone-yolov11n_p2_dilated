"""
SAHI 推理对比脚本 - 演示 SAHI 切片推理 vs 原生 YOLO 推理的效果对比

【作用】
- 演示 SAHI 切片推理和原生 YOLO 推理的对比效果
- 验证 P2 模型在高分辨率航拍图像上的微小目标检测能力
- 输出可视化结果和检测数量对比

【主要功能】
SAHI 切片推理（适合高分辨率图像，微小目标检测更好）
原生 YOLO 推理（速度快，作为对比基准）
支持 CLI 参数：灵活配置切片大小、重叠率、置信度等

【使用场景】
- 对比 SAHI 和原生推理的效果差异
- 展示微小目标检测能力
- 验证 P2 高分辨率检测头的优势
- 为实验报告生成可视化结果

【用法】
  # 使用默认配置
  python demo_inference.py
  
  # 自定义参数
  python demo_inference.py \
      --model runs/ablation/3_yolov11n_p2_dilated/weights/best.pt \
      --slice-height 640 --slice-width 640 \
      --overlap 0.2 --conf 0.25

【输出位置】
  demo_result/demo[N]_模型名/
  ├── native_yolo/              # 原生 YOLO 推理结果
  └── SAHI/                     # SAHI 切片推理结果


【特点】
- ✅ 双推理模式对比（一次运行得到两种结果）
- ✅ 输出检测数量对比，便于分析
- ✅ 支持自定义切片参数和置信度阈值
"""
import argparse
import os
import sys
import json
import random
import shutil
import glob
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO # type: ignore

# SAHI 对 YOLOv11 支持不稳定，导入失败时回退到仅原生推理
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False


def set_seed(seed: int = 42):
    """设置随机种子以保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def find_best_model(project_dir="runs/ablation"):
    """
    从消融实验汇总中寻找 mAP 最高的模型
    返回: (model_path, message)
    """
    summary_path = Path(project_dir) / "results_summary.json"
    if not summary_path.exists():
        return None, "未找到汇总文件 (results_summary.json)"

    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if not results:
            return None, "汇总文件为空"

        # 寻找 mAP@0.5:0.95 最高的实验
        # results 是个字典: {'1_baseline': {'map': 0.xxx, ...}, ...}
        best_exp = max(results.items(), key=lambda x: x[1].get('map', 0))
        best_name = best_exp[0]
        best_map = best_exp[1].get('map', 0)
        
        model_path = Path(project_dir) / best_name / "weights" / "best.pt"
        if model_path.exists():
            return str(model_path), f"自动选中最佳模型: {best_name} (mAP={best_map:.4f})"
        return None, f"最佳模型文件不存在: {model_path}"
    except Exception as e:
        return None, str(e)

def get_next_demo_dir(base_dir, model_name):
    """生成递增的输出目录，如 demo_result/demo1_modelname"""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # 提取纯模型名，去掉路径和后缀
    clean_model_name = Path(model_name).stem if Path(model_name).exists() else "unknown"
    # 如果是路径类似 runs/ablation/3_yolov11n.../weights/best.pt，尝试提取 3_yolov11n...
    try:
        if "weights" in str(model_name):
            clean_model_name = Path(model_name).parent.parent.name
    except:
        pass

    # 寻找现有的 demo 文件夹
    existing_dirs = list(base_path.glob("demo*_*"))
    max_idx = 0
    for d in existing_dirs:
        try:
            # 解析 demoN_ 中的 N
            idx = int(d.name.split('_')[0].replace('demo', ''))
            if idx > max_idx:
                max_idx = idx
        except:
            pass
    
    new_dir_name = f"demo{max_idx + 1}_{clean_model_name}"
    return base_path / new_dir_name

def parse_args():
    parser = argparse.ArgumentParser(description="SAHI vs Native YOLO Batch Inference")
    parser.add_argument("--model", default=None, help="Path to model weights (default: auto-select best)")
    parser.add_argument("--source", default="datasets/VisDrone/VisDrone2019-DET-test-dev/images", help="Path to images dir")
    parser.add_argument("--num", type=int, default=10, help="Number of random images to test")
    parser.add_argument("--output", default="demo_result", help="Base output directory")
    parser.add_argument("--slice-height", type=int, default=640, help="Slice height")
    parser.add_argument("--slice-width", type=int, default=640, help="Slice width")
    parser.add_argument("--overlap", type=float, default=0.2, help="Slice overlap")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="0", help="Device (cpu/0)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed()
    
    print("=" * 60)
    print("🚀 批量推理对比脚本 (SAHI vs Native)")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. 确定模型路径
    # ---------------------------------------------------------
    model_path = args.model
    if model_path is None:
        print("🔍 用户未指定模型，正在寻找最佳模型...")
        found_path, msg = find_best_model()
        if found_path:
            print(f"✅ {msg}")
            model_path = found_path
        else:
            # 回退到默认的 P2+Dilated 路径 (假设它存在)
            default_fallback = "runs/ablation/1_baseline_yolov11n/weights/best.pt"
            print(f"⚠️ 自动寻找失败 ({msg})")
            print(f"🔄 回退使用默认路径: {default_fallback}")
            model_path = default_fallback
            
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        print("💡 请先运行训练脚本: python ablation_study.py train all")
        return

    # ---------------------------------------------------------
    # 2. 准备图片数据
    # ---------------------------------------------------------
    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"❌ 图片目录不存在: {source_dir}")
        print("💡 请检查 VisDrone 数据集路径，或运行 convert_visdrone_to_yolo.py 确认数据")
        return
        
    # 获取目录下所有图片
    all_images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
    if not all_images:
        print(f"❌ 目录下没有找到图片: {source_dir}")
        return
        
    # 随机抽取 N 张
    num_samples = min(args.num, len(all_images))
    selected_images = random.sample(all_images, num_samples)
    print(f"📂 已从 {source_dir} 随机选中 {num_samples} 张图片进行测试")

    # ---------------------------------------------------------
    # 3. 准备输出目录
    # ---------------------------------------------------------
    # 生成如 demo_result/demo1_1_baseline_yolov11n
    out_root = get_next_demo_dir(args.output, model_path)
    
    # 创建子目录
    sahi_dir = out_root / "SAHI"
    native_dir = out_root / "native_yolo"
    
    sahi_dir.mkdir(parents=True, exist_ok=True)
    native_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 结果将保存至: {out_root}")
    print("-" * 60)

    # ---------------------------------------------------------
    # 4. 初始化模型
    # ---------------------------------------------------------
    print("🔨 加载 Native YOLO 模型...")
    try:
        yolo_model = YOLO(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 初始化 SAHI 模型
    sahi_model = None
    if SAHI_AVAILABLE:
        try:
            print("🔨 加载 SAHI 模型接口...")
            sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=model_path,
                confidence_threshold=args.conf,
                device=args.device
            )
        except Exception as e:
            print(f"⚠️ SAHI 加载失败: {e}")
    else:
        print("⚠️ 未安装 SAHI，将跳过 SAHI 推理")

    # ---------------------------------------------------------
    # 5. 循环批量推理
    # ---------------------------------------------------------
    print("\n🚀 开始批量推理...")
    for i, img_path in enumerate(selected_images):
        img_name = img_path.name
        img_stem = img_path.stem # 无后缀的文件名
        print(f"[{i+1}/{num_samples}] 处理: {img_name}")
        native_count = 0
        sahi_count = 0
        
        # --- A. Native YOLO 推理 ---
        try:
            # 使用 plot() 获取可视化结果图 (numpy array)，完全自定义保存
            # verbose=False 关闭每张图的打印刷屏
            res = yolo_model.predict(
                str(img_path), 
                conf=args.conf, 
                imgsz=640, 
                device=args.device, 
                verbose=False
            )[0]
            
            boxes = getattr(res, "boxes", None)
            native_count = len(boxes) if boxes is not None else 0
            # 绘制检测框
            im_array = res.plot()
            
            # 保存文件: native_yolo/result_xxx.jpg
            native_out_file = native_dir / f"result_{img_name}"
            cv2.imwrite(str(native_out_file), im_array)
            
        except Exception as e:
            print(f"  ❌ Native 推理出错: {e}")

        # --- B. SAHI 推理 ---
        if sahi_model:
            try:
                result = get_sliced_prediction(
                    str(img_path),
                    sahi_model,
                    slice_height=args.slice_height,
                    slice_width=args.slice_width,
                    overlap_height_ratio=args.overlap,
                    overlap_width_ratio=args.overlap,
                    verbose=0 # 关闭刷屏
                )
                sahi_count = len(getattr(result, "object_prediction_list", []) or [])
                
                # SAHI 的 export_visuals 会自动保存为 {file_name}.jpg
                # 我们先让它保存，然后重命名
                result.export_visuals(export_dir=str(sahi_dir), file_name=img_stem)
                
                # 寻找刚才生成的文件 (可能是 .jpg 或 .png)
                # SAHI 有时会改变后缀
                generated_candidates = list(sahi_dir.glob(f"{img_stem}.*"))
                
                if generated_candidates:
                    generated_file = generated_candidates[0]
                    # 重命名为 result_{原文件名}
                    # 注意保持后缀一致
                    final_name = f"result_{img_name}"
                    # 如果原图是jpg，生成了png，这里简单起见，我们保留生成文件的后缀，但文件名前缀改为 result_
                    # 比如原图 a.jpg -> 生成 a.png -> 重命名为 result_a.png
                    
                    target_file = sahi_dir / f"result_{generated_file.name}"
                    
                    # 覆盖旧文件(如果存在)
                    if target_file.exists():
                        target_file.unlink()
                        
                    generated_file.rename(target_file)
                
            except Exception as e:
                print(f"  ❌ SAHI 推理出错: {e}")

        print(f"处理完毕，原生YOLO检测到{native_count}个目标，SAHI检测到{sahi_count}个目标；")

    print("=" * 60)
    print("✅ 所有推理完成！")
    print(f"👉 结果目录: {out_root}")
    print("   ├── native_yolo/  (原生缩放推理)")
    print("   └── SAHI/         (SAHI 切片推理)")
    print("=" * 60)

if __name__ == "__main__":
    main()
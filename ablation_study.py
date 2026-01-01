"""
消融实验主脚本 - 自动化消融实验，对比 3 个模型的性能

【作用】
- 自动化消融实验的核心脚本
- 支持单独训练任意模型或一键训练全部模型
- 自动对比结果并生成汇总报告

【主要功能】
1. 单独训练：可分次训练 Baseline、P2、P2+Dilated 三个模型
2. 批量训练：一键依次训练所有 3 个模型
3. 结果对比：自动生成 mAP 对比表格和 JSON 汇总
4. 配置管理：集中管理实验配置（EXPERIMENTS 列表）

【使用场景】
- ⭐ 推荐：日常训练和消融实验的首选工具
- 适合：需要对比多个模型性能的场景
- 优势：灵活的单独训练，互不影响，随时对比

【用法】
  python ablation_study.py train 1              # 训练第1个模型 (Baseline)
  python ablation_study.py train 2              # 训练第2个模型 (P2)
  python ablation_study.py train 3              # 训练第3个模型 (P2+Dilated)
  python ablation_study.py train all            # 一键训练所有3个模型
  python ablation_study.py compare              # 对比已训练模型的结果

【输出位置】
  runs/ablation/1_baseline_yolov11n/weights/best.pt
  runs/ablation/2_yolov11n_p2/weights/best.pt
  runs/ablation/3_yolov11n_p2_dilated/weights/best.pt
  runs/ablation/results_summary.json            # 对比汇总

【特点】
- ✅ 支持灵活的单独训练（分次训练，互不影响）
- ✅ 自动生成对比表格和 JSON 汇总
- ✅ 配置集中管理，易于维护
"""
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


# 实验配置定义 (中央配置)
EXPERIMENTS = [
    {
        'id': 1,
        'name': '1_baseline_yolov11n',
        'config': 'yolo11n.yaml',
        'description': 'YOLOv11n 基线模型'
    },
    {
        'id': 2,
        'name': '2_yolov11n_p2',
        'config': 'yolov11n-p2.yaml',
        'description': 'YOLOv11n + P2 高分辨率检测头'
    },
    {
        'id': 3,
        'name': '3_yolov11n_p2_dilated',
        'config': 'yolov11n-p2-dilated.yaml',
        'description': 'YOLOv11n + P2 + 空洞深度卷积'
    }
]

EPOCHS = 100
PROJECT = "runs/ablation"


def get_experiment_by_id(exp_id):
    """根据 ID 获取实验配置"""
    for exp in EXPERIMENTS:
        if exp['id'] == exp_id:
            return exp
    return None


def train_model(config_path, experiment_name, epochs=EPOCHS, device=0):
    """
    训练单个模型配置
    
    Args:
        config_path: 模型配置文件路径
        experiment_name: 实验名称
        epochs: 训练轮数
        device: GPU 设备 ID
    """
    print("=" * 60)
    print(f"🚀 开始实验: {experiment_name}")
    print("=" * 60)
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return None
    
    # 初始化模型
    model = YOLO(config_path)
    
    # 尝试加载预训练权重
    try:
        model.load("yolo11n.pt")
        print(f"✅ 加载预训练权重: yolo11n.pt")
    except Exception as e:
        print(f"⚠️ 预训练权重加载失败: {e}")
        print("   使用随机初始化")

    # 动态调整 batch size：如果是 P2 模型(ID 2, 3)，显存压力大，用 16；否则用 32
    current_batch = 16 if ("p2" in experiment_name) else 32
    
    # 训练
    results = model.train(
        data="VisDrone.yaml",
        epochs=epochs,
        imgsz=640,
        batch=current_batch,
        device=device,
        name=experiment_name,
        project=PROJECT,
        
        # 消融实验优化
        workers=4,
        amp=True,
        cache=False,
        patience=30,  # 早停
        save_period=-1,  # 只保存最佳模型
        
        exist_ok=True,
        verbose=True,
        plots=True
    )
    
    # 返回验证集最佳指标
    best_metrics = {
        'map50': results.results_dict.get('metrics/mAP50(B)', 0),  # type: ignore
        'map': results.results_dict.get('metrics/mAP50-95(B)', 0),  # type: ignore
    }
    
    print(f"✅ {experiment_name} 训练完成")
    print(f"   mAP@0.5: {best_metrics['map50']:.4f}")
    print(f"   mAP@0.5:0.95: {best_metrics['map']:.4f}")
    
    return best_metrics

def load_experiment_results(exp_name):
    """从已完成的训练中读取结果 (修复版：支持读取 CSV)"""
    base_path = Path(PROJECT) / exp_name
    best_pt = base_path / "weights" / "best.pt"
    
    # 1. 基础检查：模型权重必须存在
    if not best_pt.exists():
        return None
    
    # 2. 优先尝试读取 YOLO 自动生成的 results.csv (最准确)
    csv_file = base_path / "results.csv"
    if csv_file.exists():
        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # 获取表头和最后一行数据
                    # YOLO 的 csv 表头和数据通常带有空格，需要 strip()
                    headers = [h.strip() for h in lines[0].split(',')]
                    values = [v.strip() for v in lines[-1].split(',')]
                    
                    # 动态查找 mAP 列的索引
                    map50_idx = -1
                    map95_idx = -1
                    
                    for i, h in enumerate(headers):
                        if 'mAP50(B)' in h:  # 查找 mAP@0.5
                            map50_idx = i
                        elif 'mAP50-95(B)' in h:  # 查找 mAP@0.5:0.95
                            map95_idx = i
                    
                    if map50_idx != -1 and map95_idx != -1:
                        return {
                            'map50': float(values[map50_idx]),
                            'map': float(values[map95_idx])
                        }
        except Exception as e:
            print(f"⚠️ 读取 CSV 失败 ({exp_name}): {e}")

    # 3. 备选方案：尝试读取 results.json (仅 train all 会生成)
    json_file = base_path / "results.json"
    if json_file.exists():
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    latest = data[-1]
                    return {
                        'map50': latest.get('metrics/mAP50(B)', 0),
                        'map': latest.get('metrics/mAP50-95(B)', 0),
                    }
        except:
            pass

    # 4. 如果读不到指标，视为结果缺失，提示补跑评估
    print(f"⚠️ 未找到 {exp_name} 的有效指标 (存在权重但缺少 results.csv/json)")
    return None


def compare_experiments():
    """对比所有已完成的实验结果"""
    print("\n" + "=" * 60)
    print("📊 消融实验结果对比")
    print("=" * 60)
    print(f"{'实验组':<30} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'提升'}")
    print("-" * 60)
    
    all_results = {}
    baseline_map50 = None
    
    for exp in EXPERIMENTS:
        exp_name = exp['name']
        metrics = load_experiment_results(exp_name)
        
        if metrics is None:
            print(f"{exp['description']:<30} ⚠️ 未完成训练")
            continue
        
        map50 = metrics['map50']
        map_full = metrics['map']
        
        if baseline_map50 is None:
            baseline_map50 = map50
            improvement = "Baseline"
        else:
            improvement = f"+{(map50 - baseline_map50) * 100:.2f}%"
        
        all_results[exp_name] = {
            'description': exp['description'],
            **metrics
        }
        
        print(f"{exp['description']:<30} {map50:<12.4f} {map_full:<15.4f} {improvement}")
    
    print("=" * 60)
    
    # 保存对比结果
    if all_results:
        os.makedirs(PROJECT, exist_ok=True)
        with open(f"{PROJECT}/results_summary.json", 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 对比结果已保存到: {PROJECT}/results_summary.json")
    
    print("\n" + "=" * 60)
    print("💡 下一步建议:")
    print("=" * 60)
    print("1. 查看训练曲线: runs/ablation/*/results.png")
    print("2. 使用最佳模型进行推理:")
    print("   python demo_inference.py")
    print("3. 详细评估:")
    print("   python eval.py --model <best_model_path>")
    print("=" * 60)


def train_single(exp_id, epochs=EPOCHS, device=0):
    """训练单个模型"""
    exp = get_experiment_by_id(exp_id)
    if not exp:
        print(f"❌ 实验 ID {exp_id} 不存在 (有效范围: 1-{len(EXPERIMENTS)})")
        return
    
    print(f"\n📋 训练实验 {exp_id}: {exp['description']}")
    print(f"⚙️  配置文件: {exp['config']}")
    print(f"📈 轮数: {epochs}")
    print(f"🖥️  设备: {device}\n")
    
    metrics = train_model(
        config_path=exp['config'],
        experiment_name=exp['name'],
        epochs=epochs,
        device=device
    )
    
    if metrics:
        print(f"\n✅ 模型 {exp_id} 训练完成")
        print(f"📁 结果位置: {PROJECT}/{exp['name']}/weights/best.pt")


def train_all(epochs=EPOCHS, device=0):
    """依次训练所有模型"""
    print("\n" + "=" * 60)
    print("🔬 消融实验: 依次训练所有模型")
    print("=" * 60)
    
    all_results = {}
    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"📋 {exp['name']}")
        print(f"{'='*60}")
        
        metrics = train_model(
            config_path=exp['config'],
            experiment_name=exp['name'],
            epochs=epochs,
            device=device
        )
        
        if metrics:
            all_results[exp['name']] = {
                'description': exp['description'],
                **metrics
            }
    
    # 输出对比结果
    print("\n" + "=" * 60)
    print("📊 消融实验结果汇总")
    print("=" * 60)
    print(f"{'实验组':<30} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'提升'}")
    print("-" * 60)
    
    baseline_map50 = None
    for name, result in all_results.items():
        map50 = result['map50']
        map_full = result['map']
        
        if baseline_map50 is None:
            baseline_map50 = map50
            improvement = "Baseline"
        else:
            improvement = f"+{(map50 - baseline_map50) * 100:.2f}%"
        
        print(f"{result['description']:<30} {map50:<12.4f} {map_full:<15.4f} {improvement}")
    
    print("=" * 60)
    
    # 保存结果到文件
    os.makedirs(PROJECT, exist_ok=True)
    with open(f'{PROJECT}/results_summary.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {PROJECT}/results_summary.json")
    print(f"📁 各实验模型位于: {PROJECT}/*/weights/best.pt")
    
    print("\n" + "=" * 60)
    print("💡 下一步建议:")
    print("=" * 60)
    print("1. 查看训练曲线对比: runs/ablation/*/results.png")
    print("2. 使用最佳模型进行 SAHI 推理:")
    print("   python demo_inference.py")
    print("3. 分析各类别 AP 变化:")
    print("   python eval.py")
    print("=" * 60)


def print_usage():
    """打印使用说明"""
    print("""
╔════════════════════════════════════════════════════════════╗
║   消融实验脚本 - YOLOv11n-P2 空洞卷积对比                    ║
╚════════════════════════════════════════════════════════════╝

用法:
  python ablation_study.py train 1              # 训练第 1 个模型 (Baseline)
  python ablation_study.py train 2              # 训练第 2 个模型 (P2)
  python ablation_study.py train 3              # 训练第 3 个模型 (P2+Dilated)
  python ablation_study.py train all            # 依次训练所有 3 个模型
  python ablation_study.py compare              # 对比已完成的所有训练结果

实验配置:
  1. Baseline YOLOv11n - 原生基线模型
  2. YOLOv11n + P2 - 高分辨率检测头
  3. YOLOv11n + P2 + Dilated - P2 + 空洞深度卷积

示例工作流:
  # 第一天: 训练 Baseline
  python ablation_study.py train 1

  # 第二天: 训练 P2 版本
  python ablation_study.py train 2

  # 第三天: 训练 P2+Dilated 版本
  python ablation_study.py train 3

  # 任何时间: 对比已完成的所有结果
  python ablation_study.py compare

  # 或一次性训练全部 (需要时间较长)
  python ablation_study.py train all
    """)


def main():
    """主程序入口"""
    if len(sys.argv) < 2:
        print_usage()
        return

    set_seed()
    
    command = sys.argv[1].lower()
    
    if command == "train":
        if len(sys.argv) < 3:
            print("❌ 缺少参数")
            print_usage()
            return
        
        target = sys.argv[2].lower()
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else EPOCHS
        device = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        
        if target == "all":
            train_all(epochs=epochs, device=device)
        else:
            try:
                exp_id = int(target)
            except ValueError:
                print(f"❌ 无效的实验 ID: {target}")
                print_usage()
                return
            
            # 🔥 关键：必须在这里调用函数，否则程序什么都不做！
            train_single(exp_id, epochs=epochs, device=device)
    
    elif command == "compare":
        compare_experiments()
    
    else:
        print(f"❌ 未知命令: {command}")
        print_usage()



if __name__ == "__main__":
    main()

import os
from ultralytics import YOLO
import argparse
import yaml

def train_yolov26(exp_name, config_path):
    """
    سكربت تدريب YOLOv26 OBB بطريقة احترافية.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # اختيار الداتا الصحيحة: exp1 بستخدم processed، والباقي بستخدم ultradata
    if exp_name == 'exp1':
        dataset_yaml = os.path.join(project_root, 'dataset', 'processed', 'data.yaml')
        print("📊 Using BASELINE dataset (processed)")
    else:
        dataset_yaml = os.path.join(project_root, 'dataset', 'ultradata', 'data.yaml')
        print("🔥 Using MODIFIED dataset (ultradata)")
    save_dir = os.path.join(project_root, 'results', 'YOLOv26')
    
    print(f"🚀 Starting YOLOv26 Training for Experiment: {exp_name}")
    print(f"📂 Dataset Config: {dataset_yaml}")
    
    # 2. قراءة الإعدادات من ملف الـ YAML الخاص بالتجربة
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        print(f"✅ Loaded config from {config_path}")
    except Exception as e:
        print(f"⚠️ Could not load config, using defaults. Error: {e}")
        cfg = {}

    # 3. تحميل الموديل (نستخدم v11n-obb كأساس لـ v26)
    model_weights = cfg.get('model', 'yolov26n-obb.pt')
    model = YOLO(model_weights)
    print(f"🏗️ Using model weights: {model_weights}")

    model.train(
        data=dataset_yaml,
        epochs=cfg.get('epochs', 100),
        imgsz=cfg.get('imgsz', 640),
        batch=cfg.get('batch', 16),
        name=exp_name,
        project=save_dir,
        device=cfg.get('device', 0),
        patience=cfg.get('patience', 50),
        save=True,
        exist_ok=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv26 OBB")
    parser.add_argument("--exp", type=str, default="exp1", help="Experiment name")
    parser.add_argument("--cfg", type=str, default=None, help="Path to config YAML")
    
    args = parser.parse_args()
    
    if args.cfg is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        args.cfg = os.path.join(project_root, 'configs', 'YOLOv26', f"{args.exp}.yaml")
    
    train_yolov26(args.exp, args.cfg)

import torch
from ultralytics import YOLO
import os

def main():
    # Print system status
    print("=" * 50)
    print("SYSTEM HEALTH CHECK")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("WARNING: CUDA not available. Running on CPU.")
        device = "cpu"
    print("=" * 50)

    # Initialize model
    # yolov11n-seg is denoted as 'yolo11n-seg.pt' in ultralytics library
    model_name = "yolo11n-seg.pt"
    print(f"Loading model: {model_name}...")
    model = YOLO(model_name)

    # Define paths
    dataset_yaml = r"C:\Users\ahmed waleed\.gemini\antigravity\scratch\structural_health_monitoring\yolo_dataset\dataset.yaml"
    project_dir = r"C:\Users\ahmed waleed\.gemini\antigravity\scratch\structural_health_monitoring\runs"
    
    # Train the model
    print("Starting training on the MDMCS dataset...")
    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=16, # Increase batch size since we have GPU memory
        device=device,
        project=project_dir,
        name="structural_damage_yolo11n_seg_100e",
        workers=2,
        plots=True
    )
    
    print("=" * 50)
    print("TRAINING FINISHED")
    print(f"Results saved in: {results.save_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()

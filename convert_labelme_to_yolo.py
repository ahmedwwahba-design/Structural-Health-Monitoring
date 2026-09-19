import os
import json
import shutil
import glob
from tqdm import tqdm

def convert_labelme_to_yolo():
    # Define source paths
    src_dir = r"C:\Users\ahmed waleed\Downloads\MDMCS A Benchmark Dataset for Multi-Damage Monitor\MDMCS A Benchmark Dataset for Multi-Damage Monitor\HRCDS\HRCDS"
    
    # Define destination path
    dest_dir = r"C:\Users\ahmed waleed\.gemini\antigravity\scratch\structural_health_monitoring\yolo_dataset"
    
    # Class mappings
    class_mapping = {
        "spalling": 0,
        "corrosion": 1,
        "exposed rebar": 2,
        "crack": 3
    }
    
    splits = ["train", "val", "test"]
    
    print("Starting dataset conversion...")
    
    for split in splits:
        print(f"Processing split: {split}")
        
        # Define directories
        img_src_dir = os.path.join(src_dir, f"{split}_image")
        ann_src_dir = os.path.join(src_dir, f"{split}_annotations")
        
        img_dest_dir = os.path.join(dest_dir, "images", split)
        lbl_dest_dir = os.path.join(dest_dir, "labels", split)
        
        # Create directories
        os.makedirs(img_dest_dir, exist_ok=True)
        os.makedirs(lbl_dest_dir, exist_ok=True)
        
        # Find all JSON files
        json_files = glob.glob(os.path.join(ann_src_dir, "*.json"))
        
        for json_file in tqdm(json_files):
            # Load JSON content
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            h = data.get("imageHeight")
            w = data.get("imageWidth")
            
            if not h or not w:
                print(f"Warning: {json_file} does not contain imageHeight or imageWidth. Skipping.")
                continue
            
            # Base name
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            
            # Locate image file
            # Labelme JSON might have imagePath. We will look for standard image formats in split_image folder
            img_file = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
                temp_img = os.path.join(img_src_dir, base_name + ext)
                if os.path.exists(temp_img):
                    img_file = temp_img
                    break
            
            if not img_file:
                print(f"Warning: Image file for {base_name} not found in {img_src_dir}. Skipping.")
                continue
            
            # Copy image to destination
            shutil.copy(img_file, os.path.join(img_dest_dir, os.path.basename(img_file)))
            
            # Output label txt file path
            label_txt_path = os.path.join(lbl_dest_dir, base_name + ".txt")
            
            with open(label_txt_path, 'w', encoding='utf-8') as out_f:
                shapes = data.get("shapes", [])
                for shape in shapes:
                    label = shape.get("label")
                    if label not in class_mapping:
                        # Skip if it is not one of our target classes
                        continue
                    
                    class_id = class_mapping[label]
                    points = shape.get("points", [])
                    
                    if not points or len(points) < 3:
                        # Polygon needs at least 3 points
                        continue
                    
                    # Normalize points
                    normalized_points = []
                    for pt in points:
                        px = pt[0] / w
                        py = pt[1] / h
                        
                        # Clip coordinates between 0 and 1
                        px = max(0.0, min(1.0, px))
                        py = max(0.0, min(1.0, py))
                        
                        normalized_points.append(f"{px:.6f} {py:.6f}")
                    
                    # Write in YOLO seg format: class_id x1 y1 x2 y2 ...
                    out_f.write(f"{class_id} " + " ".join(normalized_points) + "\n")
                    
    # Create dataset.yaml
    yaml_content = f"""path: {dest_dir.replace('\\', '/')}
train: images/train
val: images/val
test: images/test

names:
  0: spalling
  1: corrosion
  2: exposed rebar
  3: crack
"""
    
    yaml_path = os.path.join(dest_dir, "dataset.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
        
    print(f"Dataset conversion completed successfully! YOLO configuration file created at {yaml_path}")

if __name__ == "__main__":
    convert_labelme_to_yolo()

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from datetime import datetime

def generate_report(image_path, results, output_img_path, report_path):
    # Class labels
    class_names = {
        0: "spalling",
        1: "corrosion",
        2: "exposed rebar",
        3: "crack"
    }
    
    # Class colors (BGR) for visualization
    class_colors = {
        0: (0, 0, 255),      # Red for spalling
        1: (0, 165, 255),    # Orange for corrosion
        2: (42, 42, 165),    # Brown/Crimson for exposed rebar
        3: (0, 255, 255)     # Yellow for cracks
    }
    
    # Read original image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
        
    img_h, img_w, _ = img.shape
    total_pixels = img_h * img_w
    
    # Extract prediction results
    pred = results[0]
    boxes = pred.boxes
    masks = pred.masks
    
    detected_defects = []
    
    # Overlay canvas
    overlay = img.copy()
    
    if boxes is not None and len(boxes) > 0:
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = class_names.get(cls_id, f"Unknown ({cls_id})")
            color = class_colors.get(cls_id, (255, 255, 255))
            
            # Extract box coordinates
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            
            # Extract mask/polygon if available
            defect_area_px = 0
            polygon_pts = None
            if masks is not None and len(masks) > idx:
                mask_xy = masks[idx].xy[0] # List of [x, y] coordinates
                if len(mask_xy) > 0:
                    polygon_pts = mask_xy.astype(np.int32)
                    # Calculate area using OpenCV contour area
                    defect_area_px = cv2.contourArea(polygon_pts)
            
            # If mask area is 0 (or no mask), approximate using box area
            if defect_area_px == 0:
                defect_area_px = (x2 - x1) * (y2 - y1)
                
            area_pct = (defect_area_px / total_pixels) * 100
            
            # Determine severity
            # Exposed rebar is critical, so minimum severity is Medium, high if large
            if class_name == "exposed rebar":
                if area_pct > 2.0:
                    severity = "High"
                else:
                    severity = "Medium"
            elif class_name == "spalling":
                if area_pct > 3.0:
                    severity = "High"
                elif area_pct > 0.5:
                    severity = "Medium"
                else:
                    severity = "Low"
            elif class_name == "corrosion":
                if area_pct > 4.0:
                    severity = "High"
                elif area_pct > 1.0:
                    severity = "Medium"
                else:
                    severity = "Low"
            else: # crack
                if area_pct > 0.5: # Cracks are usually thin, so even small relative area is significant
                    severity = "High"
                elif area_pct > 0.1:
                    severity = "Medium"
                else:
                    severity = "Low"
                    
            detected_defects.append({
                "index": len(detected_defects) + 1,
                "class_id": cls_id,
                "type": class_name,
                "confidence": conf,
                "area_px": defect_area_px,
                "area_pct": area_pct,
                "severity": severity,
                "box": xyxy,
                "polygon": polygon_pts
            })
            
            # Draw polygon mask on overlay
            if polygon_pts is not None:
                cv2.fillPoly(overlay, [polygon_pts], color)
                cv2.polylines(img, [polygon_pts], True, color, 2)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
            # Draw bounding box on output image
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label text
            label_text = f"{class_name} ({conf:.2f}) - {severity}"
            # Put text above box
            tf = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label_text, font, 0.5, tf)[0]
            cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            cv2.putText(img, label_text, (x1, y1 - 4), font, 0.5, (255, 255, 255), tf, cv2.LINE_AA)
            
    # Apply alpha blending for mask transparency
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Save the visualized image
    cv2.imwrite(output_img_path, img)
    
    # Sort defects by severity (High first) and confidence
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    detected_defects.sort(key=lambda d: (severity_order[d["severity"]], -d["confidence"]))
    
    # Generate structural health rating
    high_count = sum(1 for d in detected_defects if d["severity"] == "High")
    med_count = sum(1 for d in detected_defects if d["severity"] == "Medium")
    low_count = sum(1 for d in detected_defects if d["severity"] == "Low")
    
    if len(detected_defects) == 0:
        overall_rating = "Excellent"
        structural_status = "No structural damage detected. The building is in excellent condition."
        recommendation = "Continue routine building envelope maintenance and annual inspections."
    elif high_count >= 2 or (high_count >= 1 and "exposed rebar" in [d["type"] for d in detected_defects if d["severity"] == "High"]):
        overall_rating = "Critical"
        structural_status = "Significant structural hazards detected (exposed reinforcement or major spalling). Structural integrity may be compromised."
        recommendation = "IMMEDIATE ACTION REQUIRED: Restrict access to the affected zones. Engage a licensed structural engineer for an emergency hands-on load assessment and design a concrete repair and cathodic protection system."
    elif high_count >= 1 or med_count >= 3:
        overall_rating = "Poor"
        structural_status = "Multiple moderate to severe defects detected. Active deterioration observed."
        recommendation = "URGENT REPAIRS NEEDED: Schedule a professional inspection within 30 days. Plan localized patch repairs, crack injection, and anti-carbonation coatings to halt progress of concrete spalling and reinforcement corrosion."
    else:
        overall_rating = "Fair"
        structural_status = "Minor surface defects detected (surface cracking, early corrosion staining, small spalls). No immediate structural hazard."
        recommendation = "MONITOR AND PLAN: Seal cracks and apply protective primers to prevent moisture ingress. Re-inspect in 6 to 12 months to monitor progression."

    # Write Markdown Report
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = f"""# Preliminary Structural Health Inspection Report
**Generated on:** {now_str}  
**Target Image:** `{os.path.basename(image_path)}`  
**Inspected By:** MDMCS AI Structural Inspection System  
**Overall Status:** **{overall_rating.upper()}**

---

## 1. Executive Summary
After analyzing the facade/structural elements from the provided inspection photograph, the AI system detected a total of **{len(detected_defects)} structural defect(s)**.

- **Overall Condition Rating:** `{overall_rating}`
- **Structural Status Description:** {structural_status}
- **Primary Recommendation:** {recommendation}

---

## 2. Defect Analysis & Breakdown
Below is the summary of defects discovered, classified by severity and confidence score:

| Defect ID | Damage Type | Confidence | Rel. Area (%) | Severity | Recommended Action |
| :---: | :--- | :---: | :---: | :---: | :--- |
"""

    action_mapping = {
        "spalling": {
            "High": "Chip loose concrete, clean corroded steel, apply zinc-rich primer, and patch with structural mortar.",
            "Medium": "Perform sound testing (hammer tap) to check for delamination, prepare surface, and patch.",
            "Low": "Clean surface and apply protective weatherproofing coating."
        },
        "corrosion": {
            "High": "Expose underlying steel, sandblast to remove rust, inspect loss of cross-section, and apply cathodic protection.",
            "Medium": "Apply rust converter/inhibitor and protective paint barrier to prevent moisture penetration.",
            "Low": "Clean rust stains and seal surface cracks."
        },
        "exposed rebar": {
            "High": "IMMEDIATE: Structural engineer must evaluate steel section loss. Replace/supplement steel bars if section loss exceeds 10%.",
            "Medium": "Clean steel rebar, apply anti-corrosive primer, and apply structural polymer-modified concrete patch.",
            "Low": "Clean steel rebar, apply anti-corrosive primer, and apply structural polymer-modified concrete patch." # Rebar is always at least Med/High
        },
        "crack": {
            "High": "Perform depth testing. Inject epoxy resin (for structural cracks) or polyurethane (for water-bearing cracks).",
            "Medium": "Route out the crack and seal with structural sealant or flexible resin.",
            "Low": "Cosmetic seal and monitor for active movement using tell-tale crack gauges."
        }
    }

    if len(detected_defects) > 0:
        for d in detected_defects:
            rec_action = action_mapping.get(d["type"], {}).get(d["severity"], "Perform localized inspection and patch repairs.")
            report_content += f"| #{d['index']} | {d['type'].title()} | {d['confidence']*100:.1f}% | {d['area_pct']:.3f}% | **{d['severity']}** | {rec_action} |\n"
    else:
        report_content += "| - | None | - | - | - | No action required. |\n"

    report_content += f"""
---

## 3. Damage Distribution Statistics
- **Total Spalling Defects:** {sum(1 for d in detected_defects if d["type"] == "spalling")}
- **Total Corrosion Defects:** {sum(1 for d in detected_defects if d["type"] == "corrosion")}
- **Total Exposed Rebar Defects:** {sum(1 for d in detected_defects if d["type"] == "exposed rebar")}
- **Total Crack Defects:** {sum(1 for d in detected_defects if d["type"] == "crack")}

---

## 4. Engineering Disclaimer
> [!CAUTION]
> **IMPORTANT NOTICE:** This report is generated by a computer vision deep learning model (`YOLOv11n-seg`) trained on concrete surface damage. It is intended solely as a preliminary decision-support tool to assist inspectors in locating and sizing defects. It does not replace a physical inspection, non-destructive testing (NDT), or engineering calculations by a certified professional structural engineer. Do not execute structural modifications based solely on this automated report.
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"Report generated successfully!")
    print(f"Annotated Image: {output_img_path}")
    print(f"Inspection Report: {report_path}")
    return detected_defects, overall_rating


def main():
    # Load model
    model_path = r"C:\Users\ahmed waleed\.gemini\antigravity\scratch\structural_health_monitoring\runs\structural_damage_yolo11n_seg_100e\weights\best.pt"

    print(f"Loading trained YOLO model from {model_path}...")
    model = YOLO(model_path)

    # Your custom image
    test_img_path = r"C:\Users\ahmed waleed\OneDrive\Desktop\OIP.webp"

    if not os.path.exists(test_img_path):
        print("Error: Image not found!")
        return

    print(f"Selecting image: {test_img_path}")

    # Output folder
    output_dir = r"C:\Users\ahmed waleed\.gemini\antigravity\scratch\structural_health_monitoring\outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Create output names based on image name
    image_name = os.path.splitext(os.path.basename(test_img_path))[0]

    output_img_path = os.path.join(
        output_dir,
        f"{image_name}_annotated.jpg"
    )

    report_path = os.path.join(
        output_dir,
        f"{image_name}_inspection_report.md"
    )

    # Run prediction
    print("Running model inference...")

    results = model.predict(
        test_img_path,
        conf=0.15,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Generate report
    defects, rating = generate_report(
        test_img_path,
        results,
        output_img_path,
        report_path
    )

    print("\n" + "="*50)
    print("INSPECTION ANALYSIS SUMMARY")
    print(f"Inspected Image: {os.path.basename(test_img_path)}")
    print(f"Overall Condition Rating: {rating}")
    print(f"Total Defects Found: {len(defects)}")

    print("Defect Details:")
    for d in defects:
        print(
            f"  - Defect #{d['index']}: {d['type']} "
            f"(Confidence: {d['confidence']*100:.1f}%, "
            f"Area Share: {d['area_pct']:.3f}%, "
            f"Severity: {d['severity']})"
        )

    print("="*50)

if __name__ == "__main__":
    main()

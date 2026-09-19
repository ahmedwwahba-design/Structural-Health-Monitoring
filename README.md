# 🏗️ Structural Health Monitoring – MDMCS

An AI-powered **Structural Health Monitoring system** designed to automatically detect and localize common types of concrete and structural damage from inspection images using **Computer Vision, Deep Learning, and YOLO-based Object Detection**.

The project uses the **MDMCS (Multi-Damage Monitoring and Classification System) dataset** to train and evaluate a deep learning model capable of identifying different structural damage categories.

## 🎯 Project Overview

Regular inspection of buildings and concrete structures is essential for detecting damage at an early stage. Traditional inspection methods often rely on manual visual examination, which can be time-consuming and may require significant effort when analyzing large numbers of images.

This project explores an automated computer vision approach that assists in the inspection process by analyzing structural images and detecting visible damage.

The system processes an input image and identifies the location and type of detected damage using bounding boxes and class labels.

### 🔍 Damage Categories

The system focuses on four major types of structural damage:

* **Crack** – Visible cracks appearing on concrete surfaces.
* **Spalling** – Areas where concrete has broken, detached, or deteriorated.
* **Corrosion** – Visible signs of corrosion-related structural deterioration.
* **Exposed Rebar** – Reinforcement bars exposed due to concrete deterioration.

## 🧠 Technologies Used

The project is implemented using:

* **Python**
* **YOLO**
* **Computer Vision**
* **Deep Learning**
* **OpenCV**
* **Ultralytics**
* **PyTorch**
* **NumPy**
* **Matplotlib**

## 📊 Dataset

The project is based on the **MDMCS Benchmark Dataset for Multi-Damage Monitoring**.

The dataset contains structural inspection images with annotations describing different types of damage.

The original annotations were converted into the **YOLO annotation format** to make them suitable for object detection training.

### Dataset Classes

| Class ID | Damage Type   |
| -------: | ------------- |
|        0 | Spalling      |
|        1 | Corrosion     |
|        2 | Exposed Rebar |
|        3 | Crack         |

The dataset is organized into three subsets:

```text
yolo_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    ├── val/
    └── test/
```

> The dataset itself is not included in this repository because of its size.

## ⚙️ Project Pipeline

The overall workflow follows these steps:

```text
Structural Inspection Images
            │
            ▼
     Dataset Preparation
            │
            ▼
   Annotation Conversion
      LabelMe → YOLO
            │
            ▼
      YOLO Dataset
            │
            ▼
      Model Training
            │
            ▼
       Model Evaluation
            │
            ▼
   Damage Detection Results
            │
            ▼
 Crack / Spalling / Corrosion /
      Exposed Rebar
```

## 📁 Project Structure

```text
Structural-Health-Monitoring/
│
├── README.md
├── train.py
├── evaluate.py
├── inspect_building.py
├── convert_labelme_to_yolo.py
├── dataset.yaml
│
├── yolo_dataset/
│   ├── images/
│   └── labels/
│
├── runs/
└── outputs/
```

> Large datasets, trained models, experiment outputs, and cache files are excluded from GitHub using `.gitignore`.

## 🔄 Annotation Conversion

The `convert_labelme_to_yolo.py` script converts the original dataset annotations into YOLO-compatible label files.

The class mapping used in the project is:

```python
class_mapping = {
    "spalling": 0,
    "corrosion": 1,
    "exposed rebar": 2,
    "crack": 3
}
```

This conversion allows the dataset to be used directly with YOLO-based object detection models.

## 🚀 Training

The `train.py` script is responsible for training the object detection model using the prepared YOLO dataset.

A typical training workflow is:

```bash
python train.py
```

The trained model and experiment results are saved locally and are excluded from the GitHub repository when they match the patterns defined in `.gitignore`.

## 📈 Evaluation

After training, the model can be evaluated using:

```bash
python evaluate.py
```

The evaluation process is used to measure the model's ability to detect and localize structural damage on unseen test images.

## 🔎 Inspection

The project also includes:

```bash
python inspect_building.py
```

This script can be used as part of the inspection workflow to analyze structural images and visualize detected damage.

## 💡 Applications

A system like this can support:

* Building condition assessment
* Concrete structure inspection
* Infrastructure monitoring
* Construction site inspection
* Preliminary structural damage screening
* Automated analysis of inspection images

The system is intended as an **AI-assisted inspection tool** and is not a replacement for professional structural engineering assessment.

## 🔮 Future Work

Potential improvements include:

* Training on a larger and more diverse dataset.
* Improving detection accuracy for small cracks.
* Adding additional structural damage categories.
* Comparing multiple YOLO architectures.
* Adding confidence-based filtering.
* Developing a web-based inspection interface.
* Supporting real-time damage detection.
* Adding visualization and reporting capabilities.
* Integrating image segmentation for more precise damage localization.

## 👨‍💻 Project

**Structural Health Monitoring – MDMCS**

Developed using **Python, Computer Vision, Deep Learning, and YOLO**.

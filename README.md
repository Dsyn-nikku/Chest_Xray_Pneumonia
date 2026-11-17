# Chest X-ray Pneumonia Classification (Educational Project)

A beginner-friendly TensorFlow/Keras image-classification project for VS Code.
It trains a small CNN to classify images into `NORMAL` and `PNEUMONIA`.

**Important:** This is an educational prototype, not a clinically validated
diagnostic system. Model scores are not calibrated diagnostic probabilities.
Do not use it to make medical decisions.

## 1. Requirements

- Windows, Python 3.11, VS Code
- CPU is supported; an NVIDIA GPU is not required
- Internet access to download Python packages and the dataset

## 2. Dataset

Download the Kaggle **Chest X-Ray Images (Pneumonia)** dataset:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Extract the archive and place its `chest_xray` directory beside these files.
The expected structure is:

```text
Chest_Xray_Pneumonia/
├── chest_xray/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── models/
├── app.py
├── train.py
├── predict.py
├── inspect_dataset.py
└── view_images.py
```

The training script creates its own 80/20 training/validation split from
`chest_xray/train`. It uses `chest_xray/test` only for final evaluation.
The original `chest_xray/val` directory is not used because it is very small
in the commonly distributed Kaggle version.

## 3. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt terminal instead:

```cmd
venv\Scripts\activate.bat
```

## 4. Install packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Check the dataset

```bash
python inspect_dataset.py
python view_images.py
```

## 6. Train the CNN

```bash
python train.py
```

The best checkpoint is saved to `models/chest_xray_model.keras`.
CPU training may take a while. Start with the default 10 epochs.

## 7. Predict one image from the terminal

Use a real image path from your dataset, for example:

```bash
python predict.py "chest_xray/test/NORMAL/IM-0001-0001.jpeg"
```

The exact filename may differ in your downloaded dataset.

## 8. Start the Streamlit application

```bash
python -m streamlit run app.py
```

Streamlit will print a local address, usually http://localhost:8501.

## Notes on evaluation and limitations

- The provided split is image-level, not patient-level. If multiple images
  from the same patient occur across splits, evaluation can be optimistic.
- Dataset labels and image acquisition conditions may contain biases.
- Accuracy alone is insufficient. Review AUC, precision, recall, confusion
  matrix, and performance across relevant subgroups.
- A clinically meaningful system would require patient-level separation,
  external validation, calibration, bias assessment, and expert oversight.

# 🎼 AI Final Project: Instrument Recognition and Dominant Instrument Detection

## 🧠 Introduction

This project aims to develop a machine learning model capable of **identifying the instruments used in a music piece** and **determining the most significant instrument**, defined as the one with the **longest cumulative playing time**.

We utilize the **[Slakh2100](https://github.com/ethman/slakh-utils)** dataset—a rich, multi-track dataset consisting of MIDI and audio (FLAC) files aligned together—to train and evaluate our model.

---

## 📊 Project Milestones

* **Initial Proposal**
  [📄 Google Doc](https://docs.google.com/document/d/1jZ7JGuw9_N2WezTFJKg-Jdab_Qe3hjmvzvFMn526SG0/edit?usp=sharing)

* **Progress Report Slides**
  [📊 Google Slides](https://docs.google.com/presentation/d/1NKNK1LOjQL-NjCYRZixUHCrWynjLjuiBDPvu7gCSuxY/edit?usp=sharing)

* **Final presentation Slides**
  [📊 Canva](https://www.canva.com/design/DAGo7rASyC4/1ifZqAWp0DwDDGjLzSdvPw/edit)
---

## 🔽 Dataset Download

Before training, you **must download the Slakh2100 dataset** from the official source:

* Download link: [https://zenodo.org/record/4038580](https://zenodo.org/record/4038580)
* Recommended version: `slakh2100_flac_redux`

After downloading, extract the dataset and place it in the following structure:

```
project_root/
├── slakh2100_flac_redux/
│   └── slakh2100_flac_redux/
│       ├── train/
│       ├── validation/
│       └── test/
├── main.py
├── test.py
└── requirements.txt
```

Make sure the dataset directory path in `main.py` matches this location. If needed, you can modify the path in your script accordingly.

---

## ⚙️ Environment Setup

### 🔧 Create a Python Virtual Environment

#### On Windows

```bash
python -m venv ai_final
ai_final\Scripts\activate
deactivate  # To exit
```

#### On macOS / Linux

```bash
python -m venv ai_final
source ai_final/bin/activate
deactivate  # To exit
```

---

## ⚡ CUDA Acceleration with PyTorch

To train with GPU acceleration, you must:

1. Install PyTorch with CUDA support.

   Go to the official installation page:
   👉 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

   Example installation (for CUDA 11.8):

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. Confirm that PyTorch detects your GPU:

   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True
   ```

3. In your training script (`main.py`), ensure the model and tensors are moved to the correct device:

   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   ```

---

## 💡 Tip

If you don’t have a GPU or compatible CUDA version, training will still run on CPU, but it will be significantly slower.

---

## ▶️ Usage

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run main pipeline:**

   ```bash
   python main.py
   ```

3. **Run test script:**

   ```bash
   python test.py
   ```

---

## 📁 Dataset Structure (Slakh2100)

```
slakh2100_flac_redux/
└── slakh2100_flac_redux/
    ├── train/
    │   ├── Track00001/
    │   │   ├── MIDI/           # MIDI files for each instrument
    │   │   ├── stems/          # Separated FLAC audio files
    │   │   ├── all_src.mid     # Combined MIDI of all tracks
    │   │   ├── metadata.yaml   # Instrument and timing info
    │   │   └── mix.flac        # Full mixed track
    │   └── ...
    ├── validation/
    │   └── Track01501/...
    ├── test/
    │   └── Track01876/...
    └── ...
```

---

## 📌 Project Goals

1. **Instrument Classification**
   Parse MIDI and metadata to label active instruments per track.

2. **Dominant Instrument Detection**
   Calculate cumulative playing time from MIDI and determine the most significant instrument.

3. **Audio/MIDI Alignment and Analysis**
   Use MIDI data for precise instrument activity segmentation.

---

## 📚 Reference

* Slakh2100 GitHub: [https://github.com/ethman/slakh-utils](https://github.com/ethman/slakh-utils)
* Original Slakh2100 Paper: [Bitton et al., 2020 (ISMIR)](https://arxiv.org/abs/2006.05261)

---

## 🚧 Future Work (Optional Section)

* Improve model generalization across unseen instrument combinations.
* Explore transformer-based models for better sequential MIDI understanding.
* Extend analysis to polyphonic dominance (multiple dominant instruments).

---

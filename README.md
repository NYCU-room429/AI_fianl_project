這是一個相當不錯的起始說明文件，以下是我幫你潤飾和補充後的版本，使其更完整、專業且更容易被讀者理解與使用。

---

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

如需加入其他部分（如 `main.py` 功能摘要、模型架構說明、成效展示等），我也可以幫忙補上。是否還要加入 README 格式、可執行範例、或 GitHub 專案介紹模版？

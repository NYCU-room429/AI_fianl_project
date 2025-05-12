# AI_fianl_project

### Slides

1. Initial Proposal: https://docs.google.com/document/d/1jZ7JGuw9_N2WezTFJKg-Jdab_Qe3hjmvzvFMn526SG0/edit?usp=sharing
2. Progress Report Slides: https://docs.google.com/presentation/d/1NKNK1LOjQL-NjCYRZixUHCrWynjLjuiBDPvu7gCSuxY/edit?usp=sharing

```
└─slakh2100_flac_redux
    └─slakh2100_flac_redux
        ├─omitted
        │  ├─Track00049
        │  │  ├─MIDI
        │  │  │  ├─...
        │  │  │  └─Sxx.mid
        │  │  ├─stems
        │  │  │  ├─...
        │  │  │  └─Sxx.flac
        │  │  ├─all_src.mid
        │  │  ├─metadata.yaml
        │  │  └─mix.flac
        │  ├─...
        │  └─Track02100
        │     ├─MIDI
        │     │  ├─...
        │     │  └─Sxx.mid
        │     ├─stems
        │     │  ├─...
        │     │  └─Sxx.flac
        │     ├─all_src.mid
        │     ├─metadata.yaml
        │     └─mix.flac
        ├─test
        │  ├─Track01876
        │  │  ├─MIDI
        │  │  │  ├─...
        │  │  │  └─Sxx.mid
        │  │  ├─stems
        │  │  │  ├─...
        │  │  │  └─Sxx.flac
        │  │  ├─all_src.mid
        │  │  ├─metadata.yaml
        │  │  └─mix.flac
        │  ├─...
        │  └─Track02098
        │     ├─MIDI
        │     │  ├─...
        │     │  └─Sxx.mid
        │     ├─stems
        │     │  ├─...
        │     │  └─Sxx.flac
        │     ├─all_src.mid
        │     ├─metadata.yaml
        │     └─mix.flac
        ├─train
        │  ├─Track00001
        │  │  ├─MIDI
        │  │  │  ├─...
        │  │  │  └─Sxx.mid
        │  │  ├─stems
        │  │  │  ├─...
        │  │  │  └─Sxx.flac
        │  │  ├─all_src.mid
        │  │  ├─metadata.yaml
        │  │  └─mix.flac
        │  ├─...
        │  └─Track01500
        │     ├─MIDI
        │     │  ├─...
        │     │  └─Sxx.mid
        │     ├─stems
        │     │  ├─...
        │     │  └─Sxx.flac
        │     ├─all_src.mid
        │     ├─metadata.yaml
        │     └─mix.flac
        └─validation
            ├─Track01501
            │ ├─MIDI
            │ │  ├─...
            │ │  └─Sxx.mid
            │ ├─stems
            │ │  ├─...
            │ │  └─Sxx.flac
            │ ├─all_src.mid
            │ ├─metadata.yaml
            │ └─mix.flac
            ├─...
            └─Track01875
            ├─MIDI
            │  ├─...
            │  └─Sxx.mid
            ├─stems
            │  ├─...
            │  └─Sxx.flac
            ├─all_src.mid
            ├─metadata.yaml
            └─mix.flac
```

### 虛擬環境

1. 建立虛擬環境

   ```bash
   python -m venv ai_final
   ai_final\Scripts\activate
   pip install -r requirements.txt
   python main.py
   ```

1. 退出虛擬環境

   ```bash
   deactivate
   ```

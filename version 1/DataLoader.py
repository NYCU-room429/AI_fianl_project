# from torch.utils.data import Dataset, DataLoader
# import torch
# import os, tqdm
# from utils import gen_melgram, get_all_class, extract_label_from_midi, get_rendered_stems

# def collect_all_path(root_path, is_test=False):
#     target_class = get_all_class('D:\\AI_fianl_project\\slakh-utils\\midi_inst_values\\general_midi_inst_0based.json')
#     midi_files = []
#     flac_files = []

#     for track in tqdm.tqdm(os.listdir(root_path), desc='Collecting all files\' path'):
#         track_path = os.path.join(root_path, track)
#         stems_dir = os.path.join(track_path, 'stems')
#         midi_dir =  os.path.join(track_path, 'MIDI')
#         metadata_path = os.path.join(track_path, 'metadata.yaml')
#         rendered_stems = get_rendered_stems(metadata_path)

#         if not os.path.isdir(track_path):
#             continue
        
#         midi_path = os.path.join(track_path, 'all_src.mid')
#         flac_path = os.path.join(track_path, 'mix.flac')
        
#         if extract_label_from_midi(midi_path, target_class) is None:
#             continue

#         midi_files.append(midi_path)
#         flac_files.append(flac_path)

#         if is_test:
#             continue
#         flac_list = sorted([f for f in os.listdir(stems_dir) if f.endswith('.flac')])
#         midi_list = sorted([f for f in os.listdir(midi_dir) if f.endswith('.mid')])

#         flac_basenames = [os.path.splitext(f)[0] for f in flac_list]
#         midi_basenames = [os.path.splitext(f)[0] for f in midi_list]
#         common_names = set(flac_basenames) & set(midi_basenames)

#         for name in sorted(common_names):
#             if name not in rendered_stems:
#                 continue
#             midi_small_path = os.path.join(midi_dir, name + '.mid')
#             flac_small_path = os.path.join(stems_dir, name + '.flac')
#             if extract_label_from_midi(midi_small_path, target_class) is None:
#                 continue
#             midi_files.append(midi_small_path)
#             flac_files.append(flac_small_path)

#     print(f"有效資料數量: {len(midi_files)}")

#     return midi_files, flac_files

# class LoadDataset(Dataset):
#     def __init__(self, midi_file_list, flac_file_list, is_train=False):
#         self.midi_file_list = midi_file_list
#         self.flac_file_list = flac_file_list
#         self.target_class = get_all_class('D:\\AI_fianl_project\\slakh-utils\\midi_inst_values\\general_midi_inst_0based.json')
#         self.is_train = is_train
#         # for midi, flac in tqdm.tqdm(zip(midi_file_list, flac_file_list), desc="init data"):
#         #     mel = gen_melgram(flac)
#         #     label = extract_label_from_midi(midi, self.target_class)
#         #     if mel is not None and label is not None:
#         #         self.midi_file_list.append(midi)
#         #         self.flac_file_list.append(flac)
#         # print(f"有效資料數量: {len(self.midi_file_list)}")

#     def __len__(self):
#         return len(self.midi_file_list)

#     def __getitem__(self, idx):
#         mel = gen_melgram(self.flac_file_list[idx], is_train = self.is_train)  # numpy array
#         mel_mean = mel.mean()
#         mel_std = mel.std()
#         if mel_std > 0:
#             mel = (mel - mel_mean) / mel_std
#         mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
#         label = extract_label_from_midi(self.midi_file_list[idx], self.target_class)
#         label_tensor = torch.tensor(label, dtype=torch.float32)  # <--- 這裡要用 float32
#         return mel_tensor, label_tensor

from torch.utils.data import Dataset, DataLoader
import torch
import os, tqdm
# 導入 joblib 模組
from joblib import Parallel, delayed

from utils import gen_melgram, get_all_class, extract_label_from_midi, get_rendered_stems

# 輔助函數：用於並行處理單個音軌目錄
def _process_single_track(track_root_path, target_class, is_test):
    local_midi_files = []
    local_flac_files = []

    track_path = track_root_path
    stems_dir = os.path.join(track_path, 'stems')
    midi_dir =  os.path.join(track_path, 'MIDI')
    metadata_path = os.path.join(track_path, 'metadata.yaml')
    
    if not os.path.isdir(track_path):
        return local_midi_files, local_flac_files # 如果不是目錄，返回空列表

    # 獲取已渲染的音軌名稱，確保 metadata.yaml 存在
    rendered_stems = []
    if os.path.exists(metadata_path):
        rendered_stems = get_rendered_stems(metadata_path)

    # 處理混合音軌 (mix.flac)
    midi_path_mix = os.path.join(track_path, 'all_src.mid')
    flac_path_mix = os.path.join(track_path, 'mix.flac')
    
    try:
        # gen_melgram 和 extract_label_from_midi 內部會自行檢查長度並返回 None
        if (os.path.exists(flac_path_mix) and os.path.exists(midi_path_mix) and
            gen_melgram(flac_path_mix, is_train=False) is not None and # 預檢查時設為 False，保持確定性
            extract_label_from_midi(midi_path_mix, target_class) is not None):
            local_midi_files.append(flac_path_mix) # 修改為先添加 flac_path
            local_flac_files.append(midi_path_mix) # 然後添加 midi_path，與返回順序一致
    except Exception: # 捕獲任何載入錯誤，跳過該檔案
        pass 

    if is_test: # 測試集只處理混合音軌
        return local_midi_files, local_flac_files

    # 處理單獨的樂器音軌 (僅限於訓練/驗證集)
    if os.path.exists(stems_dir) and os.path.exists(midi_dir): # 確保 stems 和 MIDI 目錄存在
        flac_list = sorted([f for f in os.listdir(stems_dir) if f.endswith('.flac')])
        midi_list = sorted([f for f in os.listdir(midi_dir) if f.endswith('.mid')])

        flac_basenames = [os.path.splitext(f)[0] for f in flac_list]
        midi_basenames = [os.path.splitext(f)[0] for f in midi_list]
        common_names = set(flac_basenames) & set(midi_basenames)

        for name in sorted(common_names):
            if name not in rendered_stems:
                continue
            midi_small_path = os.path.join(midi_dir, name + '.mid')
            flac_small_path = os.path.join(stems_dir, name + '.flac')
            try:
                if (os.path.exists(flac_small_path) and os.path.exists(midi_small_path) and
                    gen_melgram(flac_small_path, is_train=False) is not None and
                    extract_label_from_midi(midi_small_path, target_class) is not None):
                    local_midi_files.append(flac_small_path) # 修改為先添加 flac_path
                    local_flac_files.append(midi_small_path) # 然後添加 midi_path
            except Exception:
                pass 
    
    return local_midi_files, local_flac_files


def collect_all_path(root_path, is_test=False, n_jobs=-1): # 加入 n_jobs 參數控制並行核心數
    target_class = get_all_class('D:\\AI_fianl_project\\slakh-utils\\midi_inst_values\\general_midi_inst_0based.json')
    midi_files = [] # 這裡將用來存儲 midi_path
    flac_files = [] # 這裡將用來存儲 flac_path

    # 獲取所有可能的音軌目錄路徑
    all_tracks_to_process = [os.path.join(root_path, track_name) 
                             for track_name in os.listdir(root_path)]

    # 使用 joblib.Parallel 並行處理每個音軌目錄
    # verbose=5 會顯示進度條和每完成 5% 的任務訊息
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_single_track)(
            track_path, target_class, is_test
        ) for track_path in tqdm.tqdm(all_tracks_to_process, desc='Collecting all files\' path (Parallel)')
    )

    # 聚合所有並行任務的結果
    for local_flacs, local_midis in results: # 注意這裡的解包順序與 _process_single_track 的返回順序一致
        flac_files.extend(local_flacs)
        midi_files.extend(local_midis)

    print(f"有效資料數量: {len(midi_files)}")

    # 返回 midi_files 和 flac_files，這裡保持與原始程式碼的返回順序一致
    # 確保最終返回的 midi_files 和 flac_files 是對應的
    # 由於在 _process_single_track 和循環中，我將 flac 放在前面，那麼這裡返回的時候也應該調整
    return midi_files, flac_files # 根據 `_process_single_track` 的返回，flac_files 應該是第一個，midi_files 是第二個。
                                 # 但為了與原始程式碼的 `main.py` 和 `test.py` 接口一致，
                                 # 讓 `collect_all_path` 依然返回 `midi_files, flac_files`。
                                 # 故需要在循環中交換 extend 的順序。
                                 # 正確的做法是在 _process_single_track 中返回 (midi_list, flac_list)，
                                 # 這樣外層的 extend 就可以直接是 (midi_files.extend, flac_files.extend)。
                                 # 為了最小化修改，我在 _process_single_track 返回 (flac_list, midi_list)
                                 # 然後在外層 `for local_flacs, local_midis in results:`，
                                 # 再 `midi_files.extend(local_midis)` 和 `flac_files.extend(local_flacs)`

    # 修正後的聚合，確保返回的順序與 main.py / test.py 期望的一致
    # 為了與您現有 main.py 和 test.py 的 `train_midi_file_list, train_flac_file_list = collect_all_path(...)`
    # 這行程式碼的解包順序一致，`collect_all_path` 必須返回 `midi_files, flac_files`。
    # 所以在 `_process_single_track` 中返回的順序是 `(local_flac_files, local_midi_files)`。
    # 然後在外層的 `for local_flacs, local_midis in results:` 中，
    # `midi_files.extend(local_midis)` 和 `flac_files.extend(local_flacs)`，
    # 這樣才能讓 `midi_files` 和 `flac_files` 正確填充。

    final_midi_files = []
    final_flac_files = []
    for local_flacs, local_midis in results: # 這裡的解包順序需要與 _process_single_track 返回的順序一致
        final_flac_files.extend(local_flacs)
        final_midi_files.extend(local_midis)

    print(f"有效資料數量: {len(final_midi_files)}")
    return final_midi_files, final_flac_files # 返回 midi_files, flac_files 確保與接口一致


class LoadDataset(Dataset):
    """
    PyTorch 資料集類別，用於載入音頻和樂器標籤。
    """
    def __init__(self, midi_file_list, flac_file_list, is_train=False, num_frame=1000, segment_sec=100):
        self.midi_file_list = midi_file_list
        self.flac_file_list = flac_file_list
        self.target_class = get_all_class('D:\\AI_fianl_project\\slakh-utils\\midi_inst_values\\general_midi_inst_0based.json')
        self.is_train = is_train
        self.num_frame = num_frame
        self.segment_sec = segment_sec

    def __len__(self):
        return len(self.midi_file_list)

    def __getitem__(self, idx):
        # 載入 Mel 頻譜圖，並根據 is_train 應用數據增強
        # 確保將 num_frame 和 segment_sec 傳遞給 gen_melgram
        mel = gen_melgram(self.flac_file_list[idx], num_frame=self.num_frame, segment_sec=self.segment_sec, is_train=self.is_train)  # numpy array
        # 確保將 num_frame 和 segment_sec 傳遞給 extract_label_from_midi
        label = extract_label_from_midi(self.midi_file_list[idx], self.target_class, num_frame=self.num_frame, segment_sec=self.segment_sec) # <-- 傳遞參數
        
        # 這裡如果 mel 或 label 為 None，表示在 collect_all_path 階段的過濾不夠嚴格
        # 通常這不應該發生，如果發生，可以考慮日誌記錄並跳過該樣本或拋出錯誤
        if mel is None or label is None:
             raise ValueError(f"Failed to load data for index {idx}. Mel or label is None. FLAC: {self.flac_file_list[idx]}, MIDI: {self.midi_file_list[idx]}")

        # 數據標準化：減去均值除以標準差
        mel_mean = mel.mean()
        mel_std = mel.std()
        if mel_std > 0: # 避免除以零
            mel = (mel - mel_mean) / mel_std
        
        # 轉換為 PyTorch Tensor，並增加一個通道維度
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0) # unsqueeze(0) for channel dimension (batch, 1, n_mels, time)
        label_tensor = torch.tensor(label, dtype=torch.float32) # 標籤為浮點數，因為 BCEWithLogitsLoss 需要
        return mel_tensor, label_tensor
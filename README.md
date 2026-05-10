# 臉部去識別化、攻擊與防禦 (Face De-Identification, Its Attacks, and Defenses) - 整合版

本專案是一個整合了多位組員研究成果的小組作業，旨在評估影像「去識別化」技術對隱私與機器學習模型的影響，並探討**差分隱私 (Differential Privacy, DP)** 與**對抗性攻擊 (Adversarial Attacks)** 的相關研究。

## 整合說明
本版本以 **Ian-wu07** 的專案為基底，並整合了 **10155203rich** 關於對抗性攻擊（Adversarial Attack）的延伸研究內容。

### 主要整合內容：
1.  **傳統去識別化評估**：保留了像素化 (Pixelization)、高斯模糊 (Gaussian Blur) 與 FFT 低通濾波的評估。
2.  **對抗性攻擊 (FGSM)**：整合了分支2新增的 FGSM 攻擊機制，包含攻擊樣本生成與準確度評估。
3.  **差分隱私防禦**：保留了 Phase 3 關於拉普拉斯雜訊注入的隱私防禦研究。
4.  **視覺化增強**：Phase 1 與 Phase 2 現在會同時產出傳統混淆與對抗性樣本的對照圖。

### 程式碼修改 ：
1. **evaluate.py**： 整合了分支1的傳統去識別化評估（像素化、模糊、FFT）與分支2的 FGSM 對抗性攻擊 評估邏輯。
2. **phase1_test.py**： 修改為同時支援傳統混淆視覺化與對抗性樣本（Adversarial Samples）的生成展示。
3. **main.py**： 更新 Pipeline 流程，使其在執行時會依序完成傳統評估與對抗性攻擊評估。


## ✨ 主要功能 (Features)
- **多資料集支援**：支援 **AT&T 臉部資料庫** 與 **MNIST** 手寫數字資料集.
- **多樣化去識別化實作**：包含像素化、高斯模糊、FFT 低通濾波。
- **對抗性攻擊研究**：實作 **FGSM (Fast Gradient Sign Method)** 攻擊，評估模型在對抗性擾動下的魯棒性。
- **差分隱私 (DP) 防禦**：評估拉普拉斯雜訊在不同隱私預算 ($\epsilon$) 下的防禦效果。
- **自動化報表**：自動彙整實驗數據並產生 Markdown 報告。

## 🚀 快速開始 (Quick Start)
### 1. 安裝環境依賴
```bash
pip install -r requirements.txt
```

### 2. 執行 Pipeline
```bash
python main.py
```

### 執行流程說明：
1.  **Phase 1 (視覺化建立)**：產生傳統混淆與對抗性樣本的視覺對照圖。
2.  **Phase 2 (攻擊評估)**：評估模型在傳統混淆與對抗性攻擊下的準確度變化。
3.  **Phase 3 (DP 防禦測試)**：評估差分隱私防禦效果，計算 MSE/SSIM。
4.  **Phase 4 (報表生成)**：自動產生 `Report_Draft.md`。

## 📂 目錄結構 (Directory Structure)
```text
.
├── main.py                # Pipeline 主程式 (已整合對抗性攻擊流程)
├── data_loader.py         # PyTorch Dataset 定義
├── model.py               # 預訓練 ResNet18 模型設定
├── train.py               # 模型訓練邏輯
├── evaluate.py            # 攻擊評估邏輯 (整合傳統與對抗性攻擊)
├── deid_utils.py          # 影像混淆函式庫
├── dp_utils.py            # 差分隱私邏輯
├── metrics.py             # 評估指標工具
├── phase1_test.py         # 視覺化取樣腳本 (整合對抗性樣本)
├── phase3_test.py         # DP 防禦評估
├── generate_report.py     # 報告自動生成腳本
├── requirements.txt       # 套件依賴清單
└── results/               # 實驗結果輸出目錄 (包含 CSV, PNG, PDF 總結)
```

## 📊 整合後的輸出檔案
- **`combined_attack_results_*.csv`**：整合了傳統與對抗性攻擊的數據。
- **`phase1_traditional_*.png`**：傳統混淆視覺化。
- **`phase1_adversarial_*.png`**：對抗性樣本視覺化.
- **`phase2_traditional_samples_*.png`**：模型對傳統混淆的預測展示。
- **`phase2_adversarial_samples_*.png`**：模型對對抗性樣本的預測展示。
- **`實驗總結.pdf`**：分支2提供的額外研究總結。

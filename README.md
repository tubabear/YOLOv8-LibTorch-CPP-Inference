# YOLOv8-LibTorch-CPP-Inference

## 專案背景
嘗試在C++環境下使用已經訓練好的YOLOv8模型，此專案使用YOLOv8輸出 `.torchscript` 權重，利用 `libtorch` 來進行模型推論。

## 功能
- 利用 `libtorch` 進行YOLOv8模型推論
- 支援GPU推論

## 注意事項
- 請下載GPU版本 `libtorch`
- 使用YOLOv8提供指令生成之 `.torchscript` 是CPU版本
- 找到 `pytorch` 產生 `.torchscript` 的程式碼，把模型跟圖片轉移到 "cuda" 後再輸出就可以使用 "cuda" 版本。
  ```python
  ts = torch.jit.trace(self.model.to("cuda"), self.im.to("cuda"), strict=False)

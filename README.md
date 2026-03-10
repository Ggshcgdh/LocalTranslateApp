# 🌍 LocalTranslateApp

English ↔ Turkish translation app. Runs completely offline — internet is only needed the first time to download the model.

![Rust](https://img.shields.io/badge/Rust-1.75+-dea584?style=flat&logo=rust)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- **Local Translation**: All translation happens on your device
- **Bidirectional**: English → Turkish and Turkish → English
- **GPU Acceleration**: DirectML support for AMD/NVIDIA/Intel GPUs
- **Modern UI**: Clean and easy-to-use desktop interface

## 🚀 Installation

### Requirements

- Windows 10/11 or macOS
- Rust (1.75+)
- (Optional) GPU with DirectX 12 (Windows) or Metal (macOS)

### Build

```bash
# Clone the repository
git clone https://github.com/kodzamani/LocalTranslateApp.git
cd LocalTranslateApp

# Build and run
cargo run --release
```

## 📖 Usage

1. Launch the app
2. Select source language (English or Turkish)
3. Enter the text you want to translate
4. Click "Translate"

On first run, the model will be downloaded automatically (approximately 500MB).

## 🛠️ How It Works

- **Model**: [OPUS-MT](https://huggingface.co/Helsinki-NLP) transformer model
- **Runtime**: ONNX Runtime (with DirectML GPU acceleration)
- **UI**: Rust + eframe (egui)

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

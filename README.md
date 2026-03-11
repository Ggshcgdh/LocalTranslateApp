# 🌍 LocalTranslateApp

English ↔ Turkish translation app. Runs completely offline — internet is only needed the first time to download the model.

![Rust](https://img.shields.io/badge/Rust-1.75+-dea584?style=flat&logo=rust)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- **Local Translation**: All translation happens on your device
- **Bidirectional**: English → Turkish and Turkish → English
- **Platform Acceleration**: DirectML on Windows, CoreML on macOS
- **Modern UI**: Clean and easy-to-use desktop interface

## 🚀 Installation

### Requirements

- Windows 10/11 or macOS
- Rust (1.75+)
- (Optional) DirectX 12 GPU on Windows, or Apple GPU / Neural Engine support on macOS

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
- **Runtime**: ONNX Runtime with DirectML (Windows) or CoreML (macOS)
- **UI**: Rust + eframe (egui)

## ⚙️ Runtime Notes

- `TRANSLATE_DEVICE=cpu` forces CPU execution on every platform.
- On macOS, `TRANSLATE_COREML_UNITS=all|ane|gpu|cpu` selects the preferred CoreML compute units.
- `TRANSLATE_COREML_FORMAT=mlprogram|neuralnetwork` selects the CoreML compiled model format for debugging compatibility issues.
- `TRANSLATE_COREML_STATIC_INPUT_SHAPES=1` limits CoreML to static-shape subgraphs, which can help isolate dynamic-shape failures.
- `TRANSLATE_COREML_PROFILE_PLAN=1` asks CoreML to log which hardware each delegated operator uses.
- `TRANSLATE_COREML_DEBUG=1` prints session I/O metadata plus encoder/decoder tensor shapes and token ranges before inference.
- `TRANSLATE_COREML_CACHE=0` disables the CoreML compile cache while debugging configuration changes.
- CoreML sessions use a local compile cache under `~/Library/Caches/LocalTranslateApp/coreml` to reduce repeated startup cost.

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

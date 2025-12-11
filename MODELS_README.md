# Model Files

## ✅ Model Files Included via Git LFS

The trained model files are **included** in this repository using Git Large File Storage (LFS) to handle files larger than GitHub's 100 MB limit:

- `models/best_model.joblib` (~50 MB)
- `models/best_model_domain_adapted.joblib` (~148 MB)
- `models/preprocessor.joblib` (~5 MB)
- `models/dnn_model.pth` (~15 MB)
- `models/scaler_domain_adapted.joblib` (~5 MB)

## 📥 Getting Model Files

### Option 1: Clone Repository (Recommended)

When you clone this repository, Git LFS will automatically download the model files:

```bash
git clone https://github.com/ASK92/NIDS.git
cd NIDS
```

**Note:** Make sure Git LFS is installed on your system:
```bash
git lfs install
```

### Option 2: Train Models Yourself

Alternatively, you can train the models by running the notebook `network_ids_project.ipynb` from Section 1 through Section 8. The models will be automatically saved to the `models/` directory.

## 🚀 Quick Start

The Gradio application will automatically load models from the `models/` directory when you clone the repository.

To use the application:
1. Clone the repository (models will be downloaded via Git LFS)
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the Gradio app (Section 9 of the notebook)

## 📝 Git LFS Information

- **Total size:** ~156 MB of model files
- **Files tracked:** All `.joblib` and `.pth` files in `models/` directory
- **Git LFS status:** Active and configured

The model files are stored using Git LFS, which allows GitHub to handle files larger than 100 MB. When you clone the repository, Git LFS will automatically download these files.


# Model Files

## ⚠️ Large Model Files Excluded from Git

The trained model files are **not included** in this repository because they exceed GitHub's 100 MB file size limit:

- `models/best_model.joblib` (~50 MB)
- `models/best_model_domain_adapted.joblib` (~148 MB)
- `models/preprocessor.joblib` (~5 MB)
- `models/dnn_model.pth` (~15 MB)
- `models/scaler_domain_adapted.joblib` (~5 MB)

## 📥 How to Get Model Files

### Option 1: Train Models Yourself

Run the notebook `network_ids_project.ipynb` from Section 1 through Section 8. The models will be automatically saved to the `models/` directory.

### Option 2: Download from Release (if available)

Check the [Releases](https://github.com/ASK92/NIDS/releases) page for pre-trained model files.

### Option 3: Use Git LFS (Advanced)

If you want to include model files in the repository:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "models/*.joblib"
git lfs track "models/*.pth"

# Add and commit
git add .gitattributes
git commit -m "Add Git LFS tracking for model files"
git push
```

## 🚀 Quick Start Without Models

The Gradio application will automatically attempt to load models from the `models/` directory. If models are not found, you'll see a message prompting you to train them first.

To use the application:
1. Train models by running the notebook (Sections 1-8)
2. Models will be saved to `./models/`
3. Launch the Gradio app (Section 9)

## 📝 Note

The model files are excluded via `.gitignore` to keep the repository size manageable. The code and documentation are fully functional - you just need to train the models (or download them if available) to use the inference pipeline.


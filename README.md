# AI-Generated Video Detection Using LBP + SVM

This project builds a video classification pipeline designed to distinguish between real and AI-generated (deepfake) videos. It relies on Local Binary Pattern (LBP) for texture feature extraction, paired with a Support Vector Machine (SVM) classifier.

## Overview

- Goal: Binary video classification `real` or `not_real`
- Approach: LBP-based feature engineering combined with an SVM model.
- Core Frameworks: OpenCV, scikit-image, scikit-learn.
- Entry point:
  - Training: `train.py`
  - Single video prediction: `predict.py`

## Highlights

- A lightweight, explainable classic machine learning pipeline—perfect for research contexts.
- Texture-based feature extraction (LBP) at the frame level, seamlessly aggregated into video-level features.
- Clean separation between training and inference scripts to ensure reproducibility.

## Pipeline Flow

`video -> frame sampling -> preprocessing -> optional face crop -> LBP per frame -> agregasi fitur video -> StandardScaler -> SVM -> evaluasi`

## Core Project Structure

```text
.
|-- train.py
|-- predict.py
|-- requirements.txt
|-- README.md
|-- docs/
|   |-- generate_readme_images.ipynb
|   `-- images/
|       |-- pipeline_overview.png
|       |-- preprocessing_stages.png
|       |-- confusion_matrix.png
|       `-- sample_predictions_table.png
|-- src/
|   |-- dataset.py
|   |-- features.py
|   |-- train.py
|   |-- predict.py
|   `-- utils.py
`-- outputs/
    `-- sdfvd2_best_v3/
        |-- config.json
        |-- metrics.json (opsional)
        `-- test_predictions.csv
```

## Environment Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

Recommended dataset structure:

```text
data/
`-- sdfvd2.0/
	 |-- real/
	 `-- not_real/
```

Notes:
- The default class labels used in the code are `real` dan `not_real`.

## How to Train

Main training example:

```powershell
python train.py --dataset_root data/sdfvd2.0 --out_dir outputs/sdfvd2_best_v3 --reuse_features --svm_kernel linear --C 10 --class_weight balanced
```

Key outputs generated:
- `outputs/sdfvd2_best_v3/model.joblib`
- `outputs/sdfvd2_best_v3/config.json`
- `outputs/sdfvd2_best_v3/test_predictions.csv`
- `outputs/sdfvd2_best_v3/train.log`

## How to Predict a Single Video

```powershell
python predict.py --model outputs/sdfvd2_best_v3/model.joblib --video data/video_tes/sample.mp4
```

Main prediction results:
- Final label (`real` or `not_real`)
- Class probabilities

## Reproducibility

For consistent results:
- Keep the `random_state` fixed (default: `42`)
- Save and reuse the `config.json` from your best run.
- Make sure to use the exact same feature extraction parameters during both training and inference.

## Results & Evaluation

Primary metrics used:
- Accuracy
- Precision, Recall, F1-score per class
- Confusion matrix

Evaluation result sources:
- `test_predictions.csv`
- `metrics.json`

## Visualizations

### Pipeline Overview

![Pipeline Overview](docs/images/pipeline_overview.png)

### Preprocessing Stages

![Preprocessing Stages](docs/images/preprocessing_stages.png)

### Confusion Matrix

![Confusion Matrix](docs/images/confusion_matrix.png)

### Sample Predictions

![Sample Predictions](docs/images/sample_predictions_table.png)

## Limitations

- LBP-based approaches are quite sensitive to video quality, compression artifacts, and lighting conditions.
- Generalizing this pipeline to new datasets or domains will require additional validation.
- While classic ML models are much lighter, they generally fall short of deep learning models when trained on large-scale datasets.

## Future Roadmap

- Run benchmarks against CNN/ViT models for comparison.
- Implement cross-dataset evaluation to test generalization capabilities.
- Build a lightweight API or UI for live inference demos.

## Author

<table width="100%" style="border: none;">
  <tr style="border: none;">
    <td align="left" width="50%" style="border: none;">
      <strong>Ergy David Lundy Tumanggor</strong>
    </td>
    <td align="right" width="50%" style="border: none;">
      <a href="https://www.linkedin.com/in/ergy-david-lundy/">
        <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
      </a>
      <a href="https://github.com/Ruminas99">
        <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
      </a>
    </td>
  </tr>
</table>

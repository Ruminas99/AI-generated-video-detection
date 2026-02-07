# Deteksi AI-Generated Video (LBP + SVM)

Pipeline:
video → frame sampling → preprocessing → (optional face crop + grid) → LBP hist per frame
→ agregasi per video → StandardScaler → SVM → evaluasi → simpan model → prediksi.

## Install
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

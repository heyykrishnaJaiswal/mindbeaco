# MindBeacon — student depression risk screening (Python + Flask on Vercel)

Trained on `data/Student_Depression_Dataset.csv` (27,901 rows; 27,837 usable students).

**Accuracy (unseen test set of 5,568 students):** 85.0% accuracy, 88.7% recall, 86.1% precision, ROC-AUC 0.924. 5-fold cross-validation: 84.7% ± 0.6. Full details: `reports/accuracy_report.md`.

## Structure
- `public/index.html`, `public/bg.mp4` — the website (served by Vercel's CDN)
- `api/index.py` — Flask API: `POST /api/predict`, `GET /api/health`, `GET /api/model-info`
- `model.json` — trained model weights (loaded by the API, no ML libraries needed at runtime)
- `data/Student_Depression_Dataset.csv` — the dataset (not uploaded to Vercel)
- `train.py` — trains, evaluates and rewrites `model.json` and `reports/`
- `reports/` — `accuracy_report.md` and `metrics.json`
- `vercel.json`, `requirements.txt` — Vercel config

## Run locally
    python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python api/index.py        # open http://127.0.0.1:5000

## Retrain (optional)
    pip install -r requirements-train.txt
    python train.py data/Student_Depression_Dataset.csv   # rewrites model.json and reports/

## Deploy on Vercel
Option A, CLI:
    npm i -g vercel
    vercel login
    vercel --prod

Option B, GitHub: push this folder to a repo, then in vercel.com choose "Add New Project", import the repo, and click Deploy. No build settings are needed.

## Test the live API
    curl -X POST https://YOUR-APP.vercel.app/api/predict -H "Content-Type: application/json" \
      -d '{"age":22,"cgpa":7.5,"press":3,"sat":3,"fin":3,"hrs":6,"sleep":1,"diet":1,"fam":0,"sui":0}'

Fields: sleep 0=<5h, 1=5-6h, 2=7-8h, 3=>8h; diet 0=unhealthy, 1=moderate, 2=healthy; fam/sui 0=no, 1=yes.

## Important
This is a screening aid, not a diagnosis. Verify helpline numbers on telemanas.mohfw.gov.in before launch.

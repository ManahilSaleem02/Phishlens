#!/usr/bin/env bash
# Start the API + dashboard. Open http://localhost:8000 in your browser.
set -e
cd "$(dirname "$0")"
if [ ! -f model/phishing_model.joblib ]; then
  echo "No trained model found — training first (needs data/PhiUSIIL_Phishing_URL_Dataset.csv) ..."
  python -m model.train_model
fi
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

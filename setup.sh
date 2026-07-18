#!/usr/bin/env bash
# Maktspråk — first-time project setup
# Run from the repository root: bash setup.sh

set -euo pipefail

echo "==> Creating required directories..."
mkdir -p logs data/raw data/processed data/parquet data/models/party_classifier

echo "==> Setting up .env..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "    Copied .env.example → .env"
    echo "    Edit .env and fill in your Supabase and Twitter credentials before running the ETL."
else
    echo "    .env already exists — skipping."
fi

echo "==> Installing Python package (editable mode)..."
pip install -e ".[dev,app]"

echo ""
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  1. Fill in .env with your credentials."
echo "  2. python -m src.maktsprak_pipeline.main   # run the ETL"
echo "  3. streamlit run app/streamlit_app.py        # launch the dashboard"
echo "  4. pytest tests/ -v                          # run the test suite"

#!/usr/bin/env bash
set -e

# 1. Create & activate virtualenv
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
# pip install -r requirements.txt

# 3. Copy & fill .env
cp .env.example .env
#echo "Please edit .env and add your HF_HUB_TOKEN before continuing."

# 4. Generate 50 records
python src/merge_pipeline.py run --count 50 --outfile data/records.jsonl

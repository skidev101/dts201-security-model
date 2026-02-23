"""
main.py — Master Pipeline
============================
Run this single file to execute the complete pipeline:
  Step 1: Load & preprocess data
  Step 2: Train prescriptive model
  Step 3: Generate PDF report

Usage:
  python main.py
"""

import os, sys

# Make sure src/ is in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("=" * 60)
print("  CAMPUS SECURITY PRESCRIPTIVE MODEL — FULL PIPELINE")
print("=" * 60)

# ── Step 1 ──────────────────────────────────────────────
print("\n[STEP 1/4] Data Preprocessing")
print("-" * 40)
from src.preprocess.py import run as step1
kaggle_df = step1()

# ── Step 2 ──────────────────────────────────────────────
print("\n[STEP 2/4] Exploratory Data Analysis")
print("-" * 40)
from src.step2_eda import run as step2
stats = step2()

# ── Step 3 ──────────────────────────────────────────────
print("\n[STEP 3/4] Model Training")
print("-" * 40)
from src.step3_model import run as step3
model, rules, roc = step3()

# ── Step 4 ──────────────────────────────────────────────
print("\n[STEP 4/4] Generating PDF Report")
print("-" * 40)
from src.step4_report import build_report
if isinstance(stats, dict):
    stats["roc_auc"] = round(roc, 4)
report_path = build_report(stats=stats)

print("\n" + "=" * 60)
print("  ✅ PIPELINE COMPLETE!")
print(f"  📄 Report: {report_path}")
print(f"  📊 Plots:  outputs/plots/ ({len(os.listdir('outputs/plots'))} files)")
print(f"  🤖 Model:  outputs/models/campus_security_model.pkl")
print("=" * 60)

# Campus Security Prescriptive Model
### Development of a Prescriptive Model for Mitigating Security Challenges on University Campuses

---

## File Tree

```
campus_security_model/
│
├── data/
│   ├── raw/
│   │   ├── crime_data.csv          ← PUT YOUR KAGGLE FILE HERE
│   │   └── survey_data.csv         ← PUT YOUR GOOGLE FORM EXPORT HERE
│   └── processed/                  ← Auto-generated cleaned files
│
├── notebooks/
│   └── campus_security_model.ipynb ← Jupyter Notebook (recommended)
│
├── src/
│   ├── preprocess.py         ← Data loading & cleaning
│   ├── model.py              ← Model training & prescriptions
│
├── outputs/
│   ├── models/                     ← Saved trained model
│   └── reports/                    ← Final PDF report
│
├── main.py                      ← ONE-CLICK: runs entire pipeline
├── requirements.txt                ← Python dependencies
└── README.md                       ← This file
```

---

## Step-by-Step Setup

### Option A: Google Colab (Easiest — No Installation)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload this entire folder using the Files panel (left sidebar)
3. Open `notebooks/campus_security_model.ipynb`
4. Uncomment the `!pip install` line in the first cell and run it
5. Upload your data files when prompted
6. Click **Runtime → Run All**
7. Download your PDF report from `outputs/reports/`

---

### Option B: Local Jupyter Notebook

#### Prerequisites
- Python 3.8 or higher installed
- pip (comes with Python)

#### Step 1 — Install Python dependencies
Open a terminal / command prompt in the project folder and run:
```bash
pip install -r requirements.txt
```

#### Step 2 — Add your data files
- Copy your Kaggle CSV into: `data/raw/crime_data.csv`
- Export your Google Form (Responses → Download as CSV) into: `data/raw/survey_data.csv`

> **Note:** If you don't add these files, the model will automatically generate
> synthetic demo data so you can still see the full pipeline working.

#### Step 3 — Launch Jupyter
```bash
jupyter notebook
```
Then open `notebooks/campus_security_model.ipynb` from the browser window that opens.

#### Step 4 — Run the notebook
Click **Cell → Run All** (or press Shift+Enter on each cell)

#### Step 5 — Get your outputs
- **PDF Report:** `outputs/reports/campus_security_report.pdf`
- **Charts:** `outputs/plots/` (12 PNG files)
- **Trained Model:** `outputs/models/campus_security_model.pkl`

---

### Option C: Command Line (Fastest)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline in one command
python run_all.py
```

---

## Matching Your Survey Column Names

If your Google Form export has different column headers, open `src/step1_preprocess.py`
and find the `rename_map` dictionary (around line 95). Update the keys to match
your exact column headers. Example:

```python
rename_map = {
    "Timestamp": "TIMESTAMP",                         # leave as is
    "Age": "AGE",                                     # matches your form
    "Have you experienced a security incident...": "HAD_INCIDENT",
    # etc.
}
```

---

## What the Model Does

| Step | What Happens |
|------|-------------|
| Preprocess | Filters Kaggle data for campus premises, maps crime types to 5 categories, extracts time features |
| EDA | Generates 12 charts covering crime patterns, time analysis, survey responses |
| Model | Trains Random Forest classifier to predict high-risk incidents |
| Prescriptions | Maps model findings to specific, prioritized security recommendations |
| Report | Compiles everything into a formatted PDF research report |

---

## Troubleshooting

**"Module not found" error** → Run `pip install -r requirements.txt` again

**"File not found" for data** → The model will auto-generate demo data; or check file paths in step1_preprocess.py

**Plots not showing in notebook** → Try restarting the kernel (Kernel → Restart & Run All)

**PDF not generating** → Make sure `fpdf2` is installed: `pip install fpdf2`

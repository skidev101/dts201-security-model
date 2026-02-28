import pickle
import numpy as np
import os
import pandas as pd

# --- PATH LOGIC: CLIMB OUT OF SRC ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "campus_security_model.pkl")

def generate_security_report(hour, is_weekend, lat, lon, vict_age=20):
    """
    Loads the trained model and returns a prescriptive security report.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        print("💡 Hint: Run your Step 3 training script first!")
        return

    # Load the saved model bundle
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
        model = bundle["model"]
        features = bundle["features"]
        prescriptions = bundle["prescriptions"]

    # 1. Feature Engineering (Must match training logic exactly)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # Create a DataFrame to keep feature names consistent with the model
    input_df = pd.DataFrame([{
        "hour": hour,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "is_weekend": int(is_weekend),
        "lat": lat,
        "lon": lon,
        "vict_age": vict_age,
        "is_campus_specific": 1
    }])

    # Ensure column order matches exactly what the model saw during training
    input_df = input_df[features]

    # 2. Prediction
    risk_prob = model.predict_proba(input_df)[0][1]
    risk_level = "🔴 HIGH RISK" if risk_prob > 0.6 else "🟡 MODERATE" if risk_prob > 0.3 else "🟢 LOW"

    # 3. Prescriptive Output
    print("="*50)
    print("🛡️  CAMPUS SECURITY PRESCRIPTIVE REPORT")
    print("="*50)
    print(f"📍 Location:  ({lat}, {lon})")
    print(f"⏰ Time:      {hour:02d}:00 {'(Weekend)' if is_weekend else '(Weekday)'}")
    print(f"📊 Risk Score: {risk_prob*100:.1f}% -> {risk_level}")
    print("-" * 50)
    print("📋 RECOMMENDED ACTIONS:")

    if risk_level == "🔴 HIGH RISK":
        # Night-specific logic
        if hour >= 18 or hour <= 5:
            for p in prescriptions["HIGH_RISK_TIME_NIGHT"]:
                print(f"  {p}")
        
        # General crime prevention based on your rules
        for p in prescriptions["THEFT/ROBBERY"][:2]:
            print(f"  {p}")
        for p in prescriptions["ASSAULT/VIOLENCE"][:1]:
            print(f"  {p}")
    else:
        print("  ✅ Routine patrols sufficient.")
        print("  💡 Maintain standard 'See Something, Say Something' visibility.")
    
    print("="*50)

if __name__ == "__main__":
    # Example Test: 11 PM on a Saturday Night
    generate_security_report(hour=23, is_weekend=True, lat=34.02, lon=-118.28)
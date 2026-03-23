import os
import sys
from src.preprocess import run as step1
from src.model import run as step3
from src.report import generate_security_report

if __name__ == "__main__":

  # Make sure src/ is in the path
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

  print("=" * 60)
  print("  CAMPUS SECURITY PRESCRIPTIVE MODEL — FULL PIPELINE")
  print("=" * 60)



  # ── Step 1 ──────────────────────────────────────────────
  print("\n[STEP 1/4] Data Preprocessing")
  print("-" * 40)
  kaggle_df = step1()


  # ── Step 2 ──────────────────────────────────────────────
  # print("\n[STEP 2/4] Exploratory Data Analysis")
  # print("-" * 40)
  # from src.step2_eda import run as step2
  # stats = step2()


  # ── Step 3 ──────────────────────────────────────────────
  print("\n[STEP 3/4] Model Training")
  print("-" * 40)
  model, rules, roc = step3()

  # ── Step 4 ──────────────────────────────────────────────
  # print("\n[STEP 4/4] Generating PDF Report")
  # print("-" * 40)
  # from src.step4_report import build_report
  # if isinstance(stats, dict):
  #     stats["roc_auc"] = round(roc, 4)
  # report_path = build_report(stats=stats)

  def live_mode():
      print("\n💡 Entering Live Mode. Type 'exit' at any time to quit.")
      
      # Load bounds if available to show the user
      MODEL_PATH = "outputs/models/campus_security_model.pkl"
      bounds = None
      if os.path.exists(MODEL_PATH):
          import pickle
          with open(MODEL_PATH, "rb") as f:
              bundle = pickle.load(f)
              bounds = bundle.get("bounds")
      
      if bounds:
          print(f"📍 Active Campus Boundary:")
          print(f"   Latitude:  {bounds['lat_min']} to {bounds['lat_max']}")
          print(f"   Longitude: {bounds['lon_min']} to {bounds['lon_max']}")
      else:
          print("📍 Default Campus Boundary (UCLA):")
          print("   Latitude:  34.06 to 34.08")
          print("   Longitude: -118.46 to -118.43")

      while True:
          print("\n--- NEW ASSESSMENT ---")
          user_input = input("Enter hour (0-23) or 'exit': ").strip().lower()
          
          if user_input == 'exit':
              break
              
          try:
              hour = int(user_input)
              day_type = input("Is it a weekend? (y/n): ").strip().lower()
              is_weekend = True if day_type == 'y' else False
              
              lat = float(input("Enter Latitude: ") or 34.07)
              lon = float(input("Enter Longitude: ") or -118.44)
              
              # Run the prescriptive engine (it handles the validation internally now)
              generate_security_report(hour, is_weekend, lat, lon)
              
          except ValueError:
              print("❌ Invalid input. Please enter numbers only.")

  print("\n" + "=" * 60)
  print("  ✅ PIPELINE COMPLETE!")
  # print(f"  📄 Report: {report_path}")
  print(f"  📊 Plots:  outputs/plots/ ({len(os.listdir('outputs/plots'))} files)")
  print("  🤖 Model:  outputs/models/campus_security_model.pkl")
  print("=" * 60)

  
  # Ask the user if they want to start the live tool
  start_live = input("\nWould you like to start the Interactive Consultant? (y/n): ").lower()
  if start_live == 'y':
      live_mode()
  else: 
      print("Exiting. You can run the live mode later by executing this script again.")
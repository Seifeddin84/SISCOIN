# stage1_test.py - FIXED VERSION
"""
STAGE 1: TEST RUN (FIXED)
Tests the pipeline on first 2 subjects (24 files total).
FIXED: Increased epochs and training points for proper convergence.
"""

from balance_pinn import BalancePINNTrainer, TrainingConfig
import os
import pandas as pd
from datetime import datetime
import time

# ===========================================
# CONFIGURATION - FIXED FOR PROPER TRAINING
# ===========================================
DATA_FOLDER = r"G:\human-balance-evaluation-database-1.0.0"
OUTPUT_FOLDER = "stage1_test_results"

# Test parameters
START_FILE = 1
END_FILE = 24  # First 2 subjects (12 files each)

# FIXED: Use proper training parameters (was too short before!)
EPOCHS = 6000  # Increased from 3000
DEVICE = 'auto'
N_TRAIN_POINTS = 8000  # Increased from 5000

# ===========================================
# Setup
# ===========================================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "models"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "plots"), exist_ok=True)

config = TrainingConfig(
    epochs=EPOCHS,
    device=DEVICE,
    n_train_points=N_TRAIN_POINTS,
    enable_early_stopping=True,
    early_stopping_patience=2000  # Increased from 1000
)

def get_subject_number(file_id):
    """Get subject number from file ID."""
    return ((file_id - 1) // 12) + 1

def get_task_and_rep(file_id):
    """Get task (0-3) and repetition (0-2) from file ID."""
    position = (file_id - 1) % 12
    task = position // 3
    rep = position % 3
    return task, rep

print("="*70)
print("STAGE 1: TEST RUN (FIXED VERSION)")
print("="*70)
print(f"Data folder: {DATA_FOLDER}")
print(f"Output folder: {OUTPUT_FOLDER}")
print(f"Testing files: {START_FILE} to {END_FILE} (2 subjects, 24 files)")
print(f"Epochs: {EPOCHS} (FIXED - increased for proper training)")
print(f"Training points: {N_TRAIN_POINTS}")
print(f"Device: {config.device}")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# ===========================================
# Processing
# ===========================================
results_list = []
failed_files = []
processing_times = []
start_time = time.time()

for file_id in range(START_FILE, END_FILE + 1):
    file_name = f"BDS{file_id:05d}"
    file_path = os.path.join(DATA_FOLDER, file_name)

    subject_num = get_subject_number(file_id)
    task, rep = get_task_and_rep(file_id)

    print(f"\n[{file_id}/{END_FILE}] {file_name} (Subject {subject_num}, Task {task+1}, Rep {rep+1})")

    if not os.path.exists(file_path + ".dat"):
        print(f"  ✗ File not found")
        failed_files.append({'file_id': file_id, 'reason': 'Not found'})
        continue

    try:
        file_start = time.time()

        trainer = BalancePINNTrainer(data_path=file_path, config=config)

        print(f"  Data loaded: {len(trainer.time_data)} samples")
        print(f"  Subject: {trainer.subject_info.weight:.1f}kg, {trainer.subject_info.height:.2f}m")

        model, results = trainer.train()

        trainer.save_model(os.path.join(OUTPUT_FOLDER, "models", f"{file_name}_model.pth"))
        trainer.plot(save_path=os.path.join(OUTPUT_FOLDER, "plots", f"{file_name}.png"))

        results_dict = results.to_dict()
        results_dict.update({
            'file_id': file_id,
            'file_name': file_name,
            'subject': subject_num,
            'task': task + 1,
            'repetition': rep + 1,
            'weight_kg': trainer.subject_info.weight,
            'height_m': trainer.subject_info.height
        })

        results_list.append(results_dict)

        file_time = time.time() - file_start
        processing_times.append(file_time)

        print(f"  ✓ Success! ({file_time:.1f}s)")
        print(f"    K=[{results.stiffness_x:.1f}, {results.stiffness_y:.1f}] Nm/rad")
        print(f"    C=[{results.damping_x:.2f}, {results.damping_y:.2f}] Nm·s/rad")
        print(f"    R²=[{results.r2_x:.4f}, {results.r2_y:.4f}]")

        # WARNING if R² is too low
        if results.r2_x < 0.7 or results.r2_y < 0.7:
            print(f"  ⚠️  WARNING: Low R² - model may not be fitting well!")

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        failed_files.append({'file_id': file_id, 'reason': str(e)})
        import traceback
        traceback.print_exc()

# ===========================================
# Results & Time Estimates
# ===========================================
total_time = time.time() - start_time

print("\n" + "="*70)
print("STAGE 1 COMPLETE")
print("="*70)
print(f"Processed: {len(results_list)}/{END_FILE - START_FILE + 1}")
print(f"Failed: {len(failed_files)}")
print(f"Total time: {total_time/60:.1f} minutes")

if processing_times:
    avg_time = sum(processing_times) / len(processing_times)
    print(f"Average time per file: {avg_time:.1f}s")

    # Extrapolate to full dataset
    print("\n" + "-"*70)
    print("TIME ESTIMATES FOR FULL DATASET:")
    print("-"*70)
    print(f"Stage 2 (163 files):  ~{(163 * avg_time)/3600:.1f} hours")
    print(f"Stage 3 (1,956 files): ~{(1956 * avg_time)/3600:.1f} hours")
    print("-"*70)

# Save results
if results_list:
    df = pd.DataFrame(results_list)
    csv_path = os.path.join(OUTPUT_FOLDER, "test_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    print("\nTest Statistics:")
    print(f"  Stiffness X: {df['stiffness_x'].mean():.2f} ± {df['stiffness_x'].std():.2f} Nm/rad")
    print(f"  Stiffness Y: {df['stiffness_y'].mean():.2f} ± {df['stiffness_y'].std():.2f} Nm/rad")
    print(f"  Damping X: {df['damping_x'].mean():.3f} ± {df['damping_x'].std():.3f} Nm·s/rad")
    print(f"  Damping Y: {df['damping_y'].mean():.3f} ± {df['damping_y'].std():.3f} Nm·s/rad")
    print(f"  R² X: {df['r2_x'].mean():.4f} ± {df['r2_x'].std():.4f}")
    print(f"  R² Y: {df['r2_y'].mean():.4f} ± {df['r2_y'].std():.4f}")

    # Check if results are good
    if df['r2_x'].mean() < 0.7 or df['r2_y'].mean() < 0.7:
        print("\n" + "="*70)
        print("⚠️  WARNING: Average R² is low!")
        print("="*70)
        print("This suggests the model is not fitting well. Possible issues:")
        print("1. Data may need different preprocessing")
        print("2. Training may need more epochs")
        print("3. Check if COP data looks reasonable in plots")
        print("4. Subject parameters (weight/height) may be incorrect")
        print("\nRecommendation: Check the plots before continuing to Stage 2!")
    else:
        print("\n✓ Results look good! R² values are acceptable.")

if failed_files:
    pd.DataFrame(failed_files).to_csv(os.path.join(OUTPUT_FOLDER, "failed.csv"), index=False)

print("\n" + "="*70)
print("Review the plots in stage1_test_results/plots/")
print("If results look good, proceed to Stage 2!")
print("Run: python stage2_subjects.py")
print("="*70)

# Batch Analysis Scripts

This folder contains scripts to process the entire Human Balance Evaluation Database using the PINN model.

## Dataset Information

- **Total files**: 1,956 (BDS00001 to BDS01956)
- **Subjects**: 163
- **Structure**: Each subject has 12 files (4 tasks × 3 repetitions)

## Files in This Folder

| File | Purpose |
|------|---------|
| `run_all_stages.py` | **Master script** - Runs all stages with confirmations |
| `stage1_test.py` | Stage 1: Test on 24 files (~5-10 min) |
| `stage2_subjects.py` | Stage 2: One file per subject (163 files, ~2-3 hours) |
| `stage3_full.py` | Stage 3: All 1,956 files (~10-20 hours with GPU) |

## Quick Start

### Option 1: Run Master Script (Recommended)

The easiest way - runs all stages with confirmations:

```bash
cd batch_analysys
python run_all_stages.py
```

This will:
1. ✅ Run Stage 1 (test)
2. ⏸️ Ask if you want to continue
3. ✅ Run Stage 2 (subjects)
4. ⏸️ Ask if you want to continue
5. ✅ Run Stage 3 (full dataset)

### Option 2: Run Stages Manually

Run each stage separately:

```bash
# Stage 1: Test
python stage1_test.py

# Review results, then:
python stage2_subjects.py

# Review results, then:
python stage3_full.py
```

## Master Script Options

### Automatic Mode (No Confirmations)
```bash
python run_all_stages.py --auto
```
⚠️ **Warning**: This runs all stages without stopping!

### Skip Test Stage
```bash
python run_all_stages.py --skip-stage1
```
Goes directly to Stage 2 and 3.

### Start from Specific Stage
```bash
python run_all_stages.py --stage 2
```
Starts from Stage 2 (skips Stage 1).

### Run Specific Stages Only
```bash
python run_all_stages.py --stages "1,2"
```
Only runs Stage 1 and 2 (skips Stage 3).

## What Each Stage Does

### Stage 1: Test Run
- **Files**: 24 (first 2 subjects)
- **Time**: 5-10 minutes
- **Purpose**: Verify everything works
- **Output**: `stage1_test_results/`
- **Use when**: First time running, or after changing code

### Stage 2: One Per Subject
- **Files**: 163 (first file of each subject)
- **Time**: 2-3 hours with GPU, 8-12 hours with CPU
- **Purpose**: Get one data point per subject
- **Output**: `stage2_subjects_results/`
  - `subjects_results.csv` - Main results file
- **Use for**: Quick analysis, thesis plots, correlations

### Stage 3: Full Dataset
- **Files**: 1,956 (all trials)
- **Time**: 10-20 hours with GPU, 3-5 days with CPU
- **Purpose**: Complete dataset with all repetitions
- **Output**: `stage3_full_results/`
  - `all_1956_files_results.csv` - All trials
  - `subjects_averaged.csv` - Averaged per subject
  - `tasks_averaged.csv` - Averaged per task
- **Use for**: Final thesis analysis, task comparisons

## Output Structure

After running all stages, you'll have:

```
batch_analysys/
├── stage1_test_results/
│   ├── test_results.csv
│   ├── models/ (24 models)
│   └── plots/ (24 plots)
│
├── stage2_subjects_results/
│   ├── subjects_results.csv         ← 163 subjects
│   ├── checkpoint_subject_XXX.csv   ← Auto-saved progress
│   ├── models/ (163 models)
│   └── plots/ (163 plots)
│
└── stage3_full_results/
    ├── all_1956_files_results.csv   ← All trials
    ├── subjects_averaged.csv        ← 163 subjects (averaged)
    ├── tasks_averaged.csv           ← 652 rows (subject×task)
    ├── checkpoint_XXXXX.csv         ← Auto-saved every 50 files
    ├── failed_files.csv             ← Any failures
    ├── models/ (1,956 models)
    └── plots/ (1,956 plots)
```

## Important Features

### ✅ Auto-Save Checkpoints
- Stage 2: Saves every 25 subjects
- Stage 3: Saves every 50 files
- **You never lose progress!**

### ✅ Resume Capability
If Stage 3 gets interrupted, just run it again:
```bash
python stage3_full.py
```
It will ask if you want to resume from the last checkpoint.

### ✅ Time Estimates
All stages show:
- Current progress (%)
- Time elapsed
- Estimated time remaining
- Expected completion time (ETA)

### ✅ Failed Files Tracking
If any files fail, they're logged in `failed_files.csv` with reason.

## Recommended Workflow

### Day 1: Test & Start Subject Analysis
```bash
# Run test (10 minutes)
python run_all_stages.py --stages "1"

# Check results look good in stage1_test_results/

# Run subject analysis (2-3 hours with GPU)
python run_all_stages.py --stages "2"
```

### Day 2: Review & Start Full Analysis
```bash
# Review stage2_subjects_results/subjects_results.csv
# Check plots, statistics, etc.

# Start full dataset (leave running overnight)
python stage3_full.py
```

### Day 3: Check Results
```bash
# Check stage3_full_results/ folder
# Use the CSV files for your thesis analysis
```

## Time Estimates

### With GPU (CUDA)
- Stage 1: ~5-10 minutes
- Stage 2: ~2-3 hours
- Stage 3: ~10-20 hours
- **Total**: ~12-23 hours

### With CPU
- Stage 1: ~10-15 minutes
- Stage 2: ~8-12 hours
- Stage 3: ~3-5 days (80-120 hours)
- **Total**: ~4-5 days

💡 **Tip**: Use GPU if available! Much faster.

## Troubleshooting

### Problem: "File not found" errors
**Solution**: Check your `DATA_FOLDER` path in each script:
```python
DATA_FOLDER = r"G:\human-balance-evaluation-database-1.0.0"
```

### Problem: Stage 3 interrupted
**Solution**: Just run it again! It will resume from checkpoint:
```bash
python stage3_full.py
```

### Problem: Out of memory
**Solution**: Reduce `N_TRAIN_POINTS` in the scripts:
```python
N_TRAIN_POINTS = 5000  # Reduce from 8000
```

### Problem: Too slow
**Solutions**:
1. Check if GPU is being used (should show "cuda" not "cpu")
2. Reduce `EPOCHS` for faster (but less accurate) results:
   ```python
   EPOCHS = 3000  # Reduce from 5000
   ```
3. Use Stage 2 only (163 files instead of 1,956)

### Problem: Some files failed
**Check**: `failed_files.csv` in the output folder
- See which files and why they failed
- Common reasons: corrupted data, missing header info

## Data Format Requirements

Each file needs:
- `BDS#####.dat` - Data file with COPx and COPy columns
- `BDS#####.hea` - Header file with:
  - `#Height: XXX` (in cm)
  - `#Weight: XXX` (in kg)

## CSV Output Columns

### All Results Files
- `file_id`, `file_name` - File identifiers
- `subject_number` - Subject (1-163)
- `task`, `task_number` - Task name and number
- `repetition` - Repetition number (1-3)
- `weight_kg`, `height_m` - Subject anthropometrics
- `stiffness_x`, `stiffness_y` - Stiffness (Nm/rad)
- `damping_x`, `damping_y` - Damping (Nm·s/rad)
- `omega_n_x`, `omega_n_y` - Natural frequency (Hz)
- `zeta_x`, `zeta_y` - Damping ratio
- `rmse_x`, `rmse_y` - Model error (m)
- `r2_x`, `r2_y` - Model fit quality

## For Your Thesis

### Quick Analysis (Use Stage 2)
For correlations, basic statistics, and quick plots:
```bash
python stage2_subjects.py
```
Use: `stage2_subjects_results/subjects_results.csv`

### Complete Analysis (Use Stage 3)
For task comparisons, variability analysis, and comprehensive results:
```bash
python stage3_full.py
```
Use:
- `all_1956_files_results.csv` - All trials
- `subjects_averaged.csv` - Subject averages
- `tasks_averaged.csv` - Task comparisons

### Statistical Analysis
After processing, you can:
1. Load CSV files in Python/R
2. Calculate correlations (K vs weight, C vs height)
3. Compare tasks (ANOVA)
4. Create scatter plots
5. Perform clustering analysis

## Support

If you encounter issues:
1. Check this README
2. Review the output/error messages
3. Check `failed_files.csv` if present
4. Verify data file format and paths

## Citation

If using this for publications, cite the original PINN implementation and the Human Balance Evaluation Database.

---

**Ready to start?**

```bash
# Make sure you're in the batch_analysys folder
cd batch_analysys

# Run the master script
python run_all_stages.py
```

Good luck with your thesis! 🎓

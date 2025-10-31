# run_all_stages.py
"""
MASTER BATCH PROCESSING SCRIPT
===============================
Runs all three stages of PINN analysis with user confirmations between stages.

Stage 1: Test (24 files, ~5-10 min)
Stage 2: One per subject (163 files, ~2-3 hours)
Stage 3: Full dataset (1,956 files, ~10-20 hours)

Usage:
    python run_all_stages.py

Or with options:
    python run_all_stages.py --skip-stage1  # Skip testing, go straight to stage 2
    python run_all_stages.py --auto         # Run all stages without confirmation
"""

import subprocess
import sys
import os
from datetime import datetime
import time
import argparse

# ===========================================
# Configuration
# ===========================================
STAGE_SCRIPTS = {
    1: "stage1_test.py",
    2: "stage2_subjects.py",
    3: "stage3_full.py"
}

# Check that all stage scripts exist
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for stage, script in STAGE_SCRIPTS.items():
    script_path = os.path.join(SCRIPT_DIR, script)
    if not os.path.exists(script_path):
        print(f"❌ ERROR: {script} not found in {SCRIPT_DIR}")
        print(f"   Please make sure all stage scripts are in the same folder.")
        sys.exit(1)

# ===========================================
# Helper Functions
# ===========================================
def print_banner(text, char="="):
    """Print a banner with text."""
    print("\n" + char * 70)
    print(text.center(70))
    print(char * 70 + "\n")

def print_stage_info(stage):
    """Print information about a stage."""
    info = {
        1: {
            'name': 'STAGE 1: TEST RUN',
            'files': '24 files (2 subjects)',
            'time': '~5-10 minutes',
            'purpose': 'Validate that everything works'
        },
        2: {
            'name': 'STAGE 2: ONE PER SUBJECT',
            'files': '163 files (one per subject)',
            'time': '~2-3 hours with GPU',
            'purpose': 'Get one data point per subject'
        },
        3: {
            'name': 'STAGE 3: FULL DATASET',
            'files': '1,956 files (all trials)',
            'time': '~10-20 hours with GPU',
            'purpose': 'Complete dataset processing'
        }
    }

    s = info[stage]
    print(f"📊 {s['name']}")
    print(f"   Files: {s['files']}")
    print(f"   Time: {s['time']}")
    print(f"   Purpose: {s['purpose']}")

def run_stage(stage_num, auto_confirm=False):
    """Run a specific stage script."""
    script_name = STAGE_SCRIPTS[stage_num]
    script_path = os.path.join(SCRIPT_DIR, script_name)

    print_stage_info(stage_num)
    print(f"\n🚀 Running: {script_name}")
    print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    start_time = time.time()

    try:
        # Run the stage script
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=SCRIPT_DIR,
            check=True
        )

        elapsed = time.time() - start_time

        print("-" * 70)
        print(f"✅ Stage {stage_num} completed successfully!")
        print(f"   Time taken: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
        print(f"   End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print("-" * 70)
        print(f"❌ Stage {stage_num} failed!")
        print(f"   Error code: {e.returncode}")
        print(f"   Time before failure: {elapsed/60:.1f} minutes")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user (Ctrl+C)")
        return False

def ask_continue(stage_num, auto_confirm=False):
    """Ask user if they want to continue to next stage."""
    if auto_confirm:
        print(f"\n🤖 Auto-mode: Continuing to Stage {stage_num}...")
        time.sleep(2)
        return True

    print("\n" + "=" * 70)
    print_stage_info(stage_num)
    print("=" * 70)

    while True:
        response = input(f"\n❓ Continue to Stage {stage_num}? (y/n/review): ").lower().strip()

        if response == 'y' or response == 'yes':
            return True
        elif response == 'n' or response == 'no':
            print("⏸️  Stopping here. You can resume later by running this script again.")
            return False
        elif response == 'review' or response == 'r':
            print("\n💡 TIP: Review the output folder for the previous stage:")
            if stage_num == 2:
                print("   Check: stage1_test_results/")
            elif stage_num == 3:
                print("   Check: stage2_subjects_results/")
            print("   Look at the CSV files and some plots to verify results.\n")
        else:
            print("   Please enter 'y' (yes), 'n' (no), or 'review'")

def check_previous_results(stage_num):
    """Check if results from previous stage exist."""
    result_folders = {
        2: "stage1_test_results",
        3: "stage2_subjects_results"
    }

    if stage_num in result_folders:
        folder = result_folders[stage_num]
        folder_path = os.path.join(SCRIPT_DIR, folder)
        if os.path.exists(folder_path):
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            if csv_files:
                return True, folder

    return False, None

# ===========================================
# Main Execution
# ===========================================
def main():
    parser = argparse.ArgumentParser(
        description="Run all stages of PINN batch analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_stages.py                # Normal mode (asks for confirmation)
  python run_all_stages.py --auto         # Automatic mode (no confirmations)
  python run_all_stages.py --skip-stage1  # Skip test stage
  python run_all_stages.py --stage 2      # Start from specific stage
        """
    )

    parser.add_argument(
        '--auto',
        action='store_true',
        help='Run all stages automatically without confirmation prompts'
    )
    parser.add_argument(
        '--skip-stage1',
        action='store_true',
        help='Skip Stage 1 (test run) and go directly to Stage 2'
    )
    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2, 3],
        help='Start from specific stage (1, 2, or 3)'
    )
    parser.add_argument(
        '--stages',
        type=str,
        help='Run specific stages only (e.g., "1,2" or "2")'
    )

    args = parser.parse_args()

    # Determine which stages to run
    if args.stages:
        stages_to_run = [int(s.strip()) for s in args.stages.split(',')]
    elif args.skip_stage1:
        stages_to_run = [2, 3]
    elif args.stage:
        stages_to_run = list(range(args.stage, 4))
    else:
        stages_to_run = [1, 2, 3]

    # Print header
    print_banner("🧠 PINN BATCH ANALYSIS - MASTER SCRIPT 🧠")

    print(f"📁 Working directory: {SCRIPT_DIR}")
    print(f"🐍 Python: {sys.executable}")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Stages to run: {stages_to_run}")

    if args.auto:
        print(f"🤖 Mode: AUTOMATIC (no confirmations)")
        print("\n⚠️  WARNING: This will run all selected stages without stopping!")
        print("   Press Ctrl+C within 5 seconds to cancel...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            return
    else:
        print(f"👤 Mode: INTERACTIVE (will ask before each stage)")

    print("\n" + "=" * 70)

    # Track overall progress
    master_start_time = time.time()
    completed_stages = []

    # Run each stage
    for stage_num in stages_to_run:
        # Check if previous stage completed (if not stage 1)
        if stage_num > 1 and stage_num - 1 not in completed_stages and stage_num - 1 in stages_to_run:
            print(f"\n⚠️  Cannot run Stage {stage_num} - previous stage not completed")
            break

        # Check for existing results (except stage 1)
        if stage_num > 1:
            has_results, folder = check_previous_results(stage_num)
            if has_results:
                print(f"\n✓ Found results from previous stage in: {folder}/")
            elif stage_num - 1 in stages_to_run:
                print(f"\n⚠️  Warning: No results found from previous stage")
                if not args.auto:
                    continue_anyway = input(f"   Continue to Stage {stage_num} anyway? (y/n): ")
                    if continue_anyway.lower() != 'y':
                        print("   Skipping this stage.")
                        continue

        # Ask for confirmation (unless auto mode or first stage)
        if stage_num > 1 and not args.auto:
            if not ask_continue(stage_num, args.auto):
                break

        # Run the stage
        print_banner(f"STARTING STAGE {stage_num}", "=")

        success = run_stage(stage_num, args.auto)

        if success:
            completed_stages.append(stage_num)
            print(f"\n✅ Stage {stage_num} completed successfully!")

            # Offer to stop after each stage (unless auto mode)
            if not args.auto and stage_num < max(stages_to_run):
                print("\n💡 You can stop here and review results, or continue to next stage.")
                stop_here = input("   Stop here? (y/n): ")
                if stop_here.lower() == 'y':
                    print("⏸️  Stopping. Run this script again when ready to continue.")
                    break
        else:
            print(f"\n❌ Stage {stage_num} failed or was interrupted.")
            print(f"   You can resume by running: python {STAGE_SCRIPTS[stage_num]}")
            break

        # Small pause between stages
        if stage_num < max(stages_to_run):
            print(f"\n⏳ Pausing 5 seconds before next stage...")
            time.sleep(5)

    # Final summary
    total_time = time.time() - master_start_time

    print("\n" + "=" * 70)
    print_banner("🎉 BATCH PROCESSING SESSION COMPLETE 🎉", "=")

    print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"✅ Completed stages: {completed_stages}")
    print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Show output folders
    print("\n📂 Output folders:")
    for stage in completed_stages:
        if stage == 1:
            print("   - stage1_test_results/")
        elif stage == 2:
            print("   - stage2_subjects_results/")
        elif stage == 3:
            print("   - stage3_full_results/")

    # Next steps
    if completed_stages:
        print("\n📊 Next steps:")
        if 3 in completed_stages:
            print("   ✓ All stages complete! Ready for statistical analysis.")
            print("   ✓ Check stage3_full_results/ for:")
            print("     - all_1956_files_results.csv (all trials)")
            print("     - subjects_averaged.csv (163 subjects)")
            print("     - tasks_averaged.csv (subject × task)")
        elif 2 in completed_stages:
            print("   ✓ Stage 2 complete! You have one file per subject.")
            print("   ✓ Review stage2_subjects_results/subjects_results.csv")
            print("   → Run Stage 3 for full dataset (or run: python stage3_full.py)")
        elif 1 in completed_stages:
            print("   ✓ Stage 1 complete! Test successful.")
            print("   ✓ Review stage1_test_results/ to verify results")
            print("   → Run Stage 2 for subject data (or run: python stage2_subjects.py)")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user (Ctrl+C)")
        print("   You can resume by running this script again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

@echo off
setlocal
cd /d "%~dp0.."
if not exist "registration_runs\final_combined" mkdir "registration_runs\final_combined"
".venv\Scripts\python.exe" -u train_registration.py ^
  --output-dir registration_runs/final_combined ^
  --epochs 30 ^
  --batch-size 32 ^
  --height 96 ^
  --width 512 ^
  --base-channels 32 ^
  --max-residual-pixels 48 ^
  --synthetic-samples 4000 ^
  --num-workers 4 ^
  --patience 10 ^
  1>registration_runs\final_combined\train_stdout.log ^
  2>registration_runs\final_combined\train_stderr.log
set "TRAIN_EXIT_CODE=%ERRORLEVEL%"
if not "%TRAIN_EXIT_CODE%"=="0" (
  >registration_runs\final_combined\training_failed.txt echo Training failed with exit code %TRAIN_EXIT_CODE%.
  exit /b %TRAIN_EXIT_CODE%
)

".venv\Scripts\python.exe" evaluate_registration.py ^
  registration_runs/final_combined/best.pt ^
  --batch-size 32 ^
  --num-workers 4 ^
  --output registration_runs/final_combined/test_metrics.json ^
  1>registration_runs\final_combined\evaluation_stdout.log ^
  2>>registration_runs\final_combined\train_stderr.log
if not "%ERRORLEVEL%"=="0" (
  >registration_runs\final_combined\training_failed.txt echo IAM evaluation failed.
  exit /b 1
)

".venv\Scripts\python.exe" evaluate_cross_font.py ^
  registration_runs/final_combined/best.pt ^
  --samples 1000 ^
  --batch-size 32 ^
  --num-workers 4 ^
  --output registration_runs/final_combined/cross_font_metrics.json ^
  1>registration_runs\final_combined\cross_font_stdout.log ^
  2>>registration_runs\final_combined\train_stderr.log
if not "%ERRORLEVEL%"=="0" (
  >registration_runs\final_combined\training_failed.txt echo Cross-font evaluation failed.
  exit /b 1
)

".venv\Scripts\python.exe" visualize_registration.py ^
  registration_runs/final_combined/best.pt ^
  --output-dir alignment_output/final_examples ^
  --count 8 ^
  1>registration_runs\final_combined\visualization_stdout.log ^
  2>>registration_runs\final_combined\train_stderr.log
if not "%ERRORLEVEL%"=="0" (
  >registration_runs\final_combined\training_failed.txt echo Visualization failed.
  exit /b 1
)

>registration_runs\final_combined\training_complete.txt echo Training and evaluation completed successfully.
exit /b 0

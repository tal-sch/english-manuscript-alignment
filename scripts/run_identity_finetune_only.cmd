@echo off
setlocal
cd /d "%~dp0.."
if not exist "registration_runs\identity_finetune" mkdir "registration_runs\identity_finetune"

".venv\Scripts\python.exe" -u train_registration.py ^
  --output-dir registration_runs/identity_finetune ^
  --epochs 8 ^
  --batch-size 32 ^
  --learning-rate 0.00005 ^
  --height 96 ^
  --width 512 ^
  --base-channels 32 ^
  --max-residual-pixels 48 ^
  --synthetic-samples 4000 ^
  --identity-probability 0.20 ^
  --num-workers 4 ^
  --patience 8 ^
  --init-checkpoint registration_runs/final_combined/best.pt ^
  1>registration_runs\identity_finetune\train_stdout.log ^
  2>registration_runs\identity_finetune\train_stderr.log
if errorlevel 1 goto identity_failed

for %%C in (best last) do (
  ".venv\Scripts\python.exe" evaluate_registration.py registration_runs/identity_finetune/%%C.pt ^
    --batch-size 32 --num-workers 4 --output registration_runs/identity_finetune/%%C_test_metrics.json ^
    1>registration_runs\identity_finetune\%%C_evaluation_stdout.log 2>>registration_runs\identity_finetune\train_stderr.log
  if errorlevel 1 goto identity_failed
  ".venv\Scripts\python.exe" evaluate_cross_font.py registration_runs/identity_finetune/%%C.pt ^
    --samples 1000 --batch-size 32 --num-workers 4 ^
    --output registration_runs/identity_finetune/%%C_cross_font_metrics.json ^
    1>registration_runs\identity_finetune\%%C_cross_font_stdout.log 2>>registration_runs\identity_finetune\train_stderr.log
  if errorlevel 1 goto identity_failed
  ".venv\Scripts\python.exe" evaluate_identity.py registration_runs/identity_finetune/%%C.pt ^
    --batch-size 32 --output registration_runs/identity_finetune/%%C_identity_metrics.json ^
    1>registration_runs\identity_finetune\%%C_identity_stdout.log 2>>registration_runs\identity_finetune\train_stderr.log
  if errorlevel 1 goto identity_failed
  ".venv\Scripts\python.exe" evaluate_real_pairs.py registration_runs/identity_finetune/%%C.pt ^
    --output-dir alignment_output/real_pairs_identity_%%C ^
    1>registration_runs\identity_finetune\%%C_real_pairs_stdout.log 2>>registration_runs\identity_finetune\train_stderr.log
  if errorlevel 1 goto identity_failed
)

".venv\Scripts\python.exe" visualize_registration.py registration_runs/identity_finetune/best.pt ^
  --output-dir alignment_output/identity_finetune_examples --count 8 ^
  1>registration_runs\identity_finetune\visualization_stdout.log ^
  2>>registration_runs\identity_finetune\train_stderr.log
if errorlevel 1 goto identity_failed

>registration_runs\identity_finetune\training_complete.txt echo Identity fine-tuning and evaluation completed successfully.
>registration_runs\improvement_experiments_complete.txt echo All improvement experiments completed successfully.
exit /b 0

:identity_failed
>registration_runs\identity_finetune\training_failed.txt echo Identity fine-tuning experiment failed.
exit /b 1

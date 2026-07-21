@echo off
setlocal
cd /d "%~dp0.."

if not exist "registration_runs\iam_only_30" mkdir "registration_runs\iam_only_30"
if not exist "registration_runs\identity_finetune" mkdir "registration_runs\identity_finetune"

".venv\Scripts\python.exe" -u train_registration.py ^
  --output-dir registration_runs/iam_only_30 ^
  --epochs 30 ^
  --batch-size 32 ^
  --height 96 ^
  --width 512 ^
  --base-channels 32 ^
  --max-residual-pixels 48 ^
  --synthetic-samples 0 ^
  --identity-probability 0 ^
  --num-workers 4 ^
  --patience 10 ^
  1>registration_runs\iam_only_30\train_stdout.log ^
  2>registration_runs\iam_only_30\train_stderr.log
if errorlevel 1 goto iam_failed

".venv\Scripts\python.exe" evaluate_registration.py registration_runs/iam_only_30/best.pt ^
  --batch-size 32 --num-workers 4 --output registration_runs/iam_only_30/test_metrics.json ^
  1>registration_runs\iam_only_30\evaluation_stdout.log 2>>registration_runs\iam_only_30\train_stderr.log
if errorlevel 1 goto iam_failed
".venv\Scripts\python.exe" evaluate_cross_font.py registration_runs/iam_only_30/best.pt ^
  --samples 1000 --batch-size 32 --num-workers 4 ^
  --output registration_runs/iam_only_30/cross_font_metrics.json ^
  1>registration_runs\iam_only_30\cross_font_stdout.log 2>>registration_runs\iam_only_30\train_stderr.log
if errorlevel 1 goto iam_failed
".venv\Scripts\python.exe" evaluate_identity.py registration_runs/iam_only_30/best.pt ^
  --batch-size 32 --output registration_runs/iam_only_30/identity_metrics.json ^
  1>registration_runs\iam_only_30\identity_stdout.log 2>>registration_runs\iam_only_30\train_stderr.log
if errorlevel 1 goto iam_failed
".venv\Scripts\python.exe" evaluate_real_pairs.py registration_runs/iam_only_30/best.pt ^
  --output-dir alignment_output/real_pairs_iam_only ^
  1>registration_runs\iam_only_30\real_pairs_stdout.log 2>>registration_runs\iam_only_30\train_stderr.log
if errorlevel 1 goto iam_failed
>registration_runs\iam_only_30\training_complete.txt echo IAM-only training and evaluation completed successfully.

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

:iam_failed
>registration_runs\iam_only_30\training_failed.txt echo IAM-only experiment failed.
exit /b 1

:identity_failed
>registration_runs\identity_finetune\training_failed.txt echo Identity fine-tuning experiment failed.
exit /b 1

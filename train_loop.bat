@echo off
setlocal enabledelayedexpansion

title Pokémon Red Training - Automatic Restart with Dynamic Checkpoint
echo Starting training loop...

:inicio
echo --------------------------
echo Checking for previous checkpoint...
echo --------------------------

set "CHECKPOINT_PATH="

REM If a checkpoint path file exists, read the path
if exist last_checkpoint_path.txt (
    set /p CHECKPOINT_PATH=<last_checkpoint_path.txt
    echo Checkpoint found: !CHECKPOINT_PATH!
) else (
    echo No previous checkpoint found.
)

echo --------------------------
echo Running training...
echo --------------------------

REM Run training script with checkpoint path if available
if defined CHECKPOINT_PATH (
    python train.py "!CHECKPOINT_PATH!"
) else (
    python train.py
)

echo.
echo Training finished. Restarting in 5 seconds...
timeout /t 5 >nul
goto inicio

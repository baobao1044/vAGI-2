@echo off
chcp 65001 >nul 2>&1
title vAGI-2 Training (Small Model)
cd /d "%~dp0"

echo.
echo  ==========================================
echo   vAGI-2 Vietnamese Training
echo  ==========================================
echo.
echo  Step 1: Download data (if needed)
echo  Step 2: Train model
echo.

if not exist "data\vi_sentences.txt" (
    echo [!] No data found. Downloading...
    powershell -ExecutionPolicy Bypass -File scripts\download_vi_large.ps1
)

echo.
echo === Starting training (small model, 20 epochs) ===
echo === This will take several hours, you can leave it running ===
echo.
cargo run --example train_vietnamese -p vagi-lm --release -- --small --epochs 20
echo.
echo === Training complete! ===
echo Run chat: cargo run --example chat_vi -p vagi-lm --release
pause

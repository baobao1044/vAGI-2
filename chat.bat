@echo off
chcp 65001 >nul 2>&1
title vAGI-2 Vietnamese Chat
cd /d "%~dp0"
cargo run --example chat_vi -p vagi-lm --release
pause

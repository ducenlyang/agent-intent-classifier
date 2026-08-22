@echo off
chcp 65001 >nul
cd /d %~dp0
echo === 教育助手解决方案 · 一键启动 ===
.venv\Scripts\python.exe run_demo.py
pause

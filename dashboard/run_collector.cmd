@echo off
rem Runs the Kalshi live order-book collector (scheduled task KalshiLiveCollector).
cd /d "%~dp0"
"C:\Users\Lenovo\miniconda3\envs\research_py310\python.exe" live_collector.py >> "..\data\live\collector.log" 2>&1

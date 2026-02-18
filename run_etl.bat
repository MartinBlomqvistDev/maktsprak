@echo off
REM MaktsprakAI — scheduled ETL runner
REM
REM Uses %%~dp0 (the directory containing this script) instead of a hard-coded
REM path so the script works on any machine or user account without modification.
REM
REM Set up in Windows Task Scheduler:
REM   Action:   Start a program
REM   Program:  C:\path\to\MaktsprakAI\run_etl.bat
REM   Start in: (leave blank — the script handles its own working directory)

SET "PROJ=%~dp0"

REM Change to the project root so Python can find src/
cd /d "%PROJ%"

REM Activate virtual environment
call "%PROJ%.venv\Scripts\activate.bat"

REM Run the incremental ETL — output goes to logs/ alongside the timestamped loguru files
python -m src.maktsprak_pipeline.main >> "%PROJ%logs\etl_scheduled.log" 2>&1

REM Deactivate
deactivate

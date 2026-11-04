@echo off
echo Starting Bank Muscat FastAPI Real-Time Scoring Service...
call .\venv\Scripts\activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
pause

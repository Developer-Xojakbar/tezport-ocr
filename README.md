```bash
# Packages create + activate
python3 -m venv .venv
source .venv/bin/activate

# Packages install
pip install -r requirements.txt

# Run Project
python main.py

# Run Project RENDER
uvicorn main:app --host 0.0.0.0 --port $PORT
```

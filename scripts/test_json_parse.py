import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env_load import load_dotenv
from src.llm_extract import extract_user_request, parse_first_json_object

load_dotenv()

samples = [
    '{"status":"ready","country":"GBRTENW","gender":"male",}',
    'Sure: {"status": "ready", "country": "SWE", "gender": "male", "models": ["LC",]}',
    '{"status":"ready","country":"USA","gender":"female"',
]
for s in samples:
    try:
        print("OK", parse_first_json_object(s))
    except Exception as e:
        print("FAIL", type(e).__name__, e)

r = extract_user_request(
    "Predict mortality for GBRTENW male, ages 50-89, fit 1960-1990, "
    "test 1991-2015, models LC RH, mse only."
)
print("EXTRACT", r)

from flask import Flask, request, jsonify
import json
import re
import parsedatetime
from datetime import datetime, timedelta
import dateutil.parser
from openai import OpenAI


app = Flask(__name__)


from dotenv import load_dotenv
import os

# Load environment variables from .env (for local dev)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in environment variables or .env file.")

# Initialize OpenAI client
# This supports both normal keys (sk-...) and project keys (sk-proj-...)
client = OpenAI(
    api_key=OPENAI_API_KEY,
    organization=OPENAI_ORG_ID if OPENAI_ORG_ID else None,
    project=OPENAI_PROJECT_ID if OPENAI_PROJECT_ID else None
)

def build_prompt(query):
    return f"""
You are a multilingual flight booking assistant.

The current date is {datetime.now().strftime('%Y-%m-%d')}.
The user query may be in Hindi, English, or a mix of both.

You must:
- Understand Hindi date/time phrases (e.g., "कल", "परसों", "5 दिन बाद", "अगला सोमवार", "अगले महीने") the same as their English equivalents ("tomorrow", "day after tomorrow", "after 5 days", "next Monday", "next month").
- Always calculate `depdate` relative to today's date.
- If `retdate` is a relative expression (e.g., "10 दिन बाद", "after 10 days"), calculate it relative to the `depdate` instead of today's date.
- Return all dates in YYYY-MM-DD format in the JSON output.
- Handle both Hindi and English city/airport names and convert them to their **IATA 3-letter codes**.
- If a location has multiple airports, choose the primary international passenger airport.

Extract and return only JSON with the following keys:
- from: departure airport IATA code (3 letters, e.g., DEL for Delhi)
- to: arrival airport IATA code (3 letters, e.g., DXB for Dubai)
- depdate: departure date in YYYY-MM-DD format
- retdate: return date in YYYY-MM-DD format (optional)
- adults: number of adults (default: 1)
- children: number of children (default: 0)
- infants: number of infants (default: 0)
- cabin: cabin class like economy, business (default: economy)
- airline_include: preferred airline if mentioned (e.g., "by Indigo", "इंडिगो से")

Rules:
- Only assign a value if it is clearly mentioned in the query.
- If a field is missing, set its value to null or "Not Provided", except:
  * Set "adults" to 1 by default
  * Set "children" and "infants" to 0 by default
  * Set "cabin" to "economy" by default

Return valid JSON only. Do not explain anything.

Query: "{query}"
"""

from dateutil.relativedelta import relativedelta  # put at top of file

def parse_date_string(natural_date, base_date=None):
    if not natural_date:
        return None

    natural_date_lower = natural_date.lower().strip()
    today = base_date or datetime.now()

    # Special cases first
    if "day after tomorrow" in natural_date_lower:
        return (today + timedelta(days=2)).strftime('%Y-%m-%d')
    if "tomorrow" in natural_date_lower:
        return (today + timedelta(days=1)).strftime('%Y-%m-%d')

    # Handle "next month"
    if natural_date_lower == "next month":
        return (today + relativedelta(months=1)).strftime('%Y-%m-%d')

    match = re.search(r'after (\d+) days?', natural_date_lower)
    if match:
            days = int(match.group(1))
            return (today + timedelta(days=days)).strftime('%Y-%m-%d')

    # parsedatetime fallback
    cal = parsedatetime.Calendar()
    time_struct, parse_status = cal.parse(natural_date, sourceTime=today.timetuple())
    if parse_status != 0:
        return datetime(*time_struct[:6]).strftime('%Y-%m-%d')

    # dateutil fallback
    try:
        return dateutil.parser.parse(natural_date, fuzzy=True, default=today).strftime('%Y-%m-%d')
    except Exception:
        return None


def is_missing(value):
    if value is None:
        return True
    value = str(value).strip().lower()
    return value in ["", "none", "not provided", "departure city (not provided)", "arrival city (not provided)"]

@app.route("/", methods=["GET"])
def home():
    return """
    <h1>✈️ Flight Booking Chatbot API</h1>
    <p>Welcome! This API extracts flight details from natural language queries.</p>
    """

@app.route('/search', methods=['POST'])
def parse_query():
    try:
        data = request.get_json()
        user_query = data.get("query")
        if not user_query:
            return jsonify({"error": "Missing 'query'"}), 400

        # Build the prompt
        prompt = build_prompt(user_query)

        # Call GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            return jsonify({"error": "Invalid JSON in model output"}), 500

        parsed = json.loads(json_match.group())

        depdate_raw = parsed.get("depdate")
        retdate_raw = parsed.get("retdate")
        depfrom = parsed.get("from")
        arrto = parsed.get("to")

        depdate = parse_date_string(depdate_raw) if depdate_raw else None
        retdate = parse_date_string(retdate_raw, base_date=datetime.strptime(depdate, "%Y-%m-%d")) if retdate_raw and depdate else None

        adults = int(parsed.get("adults", 1))
        children = int(parsed.get("children", 0))
        infants = int(parsed.get("infants", 0))
        cabin = parsed.get("cabin", "economy").lower()
        airline = parsed.get("airline_include", "")

        missing_fields = []
        follow_up_questions = []

        if is_missing(depfrom):
            missing_fields.append("from")
            follow_up_questions.append("✈️ Where are you flying *from*?")
        if is_missing(arrto):
            missing_fields.append("to")
            follow_up_questions.append("🛬 Where are you flying *to*?")
        if not depdate_raw or not depdate:
            missing_fields.append("depdate")
            follow_up_questions.append("📅 When do you want to *depart*?")

        if missing_fields:
            return jsonify({
                "status": "incomplete",
                "message": f"Missing fields: {', '.join(missing_fields)}",
                "missing_fields": missing_fields,
                "follow_up": follow_up_questions,
                "parsed": {
                    "from": depfrom,
                    "to": arrto,
                    "depdate": depdate_raw or None
                }
            })

        payload = {
            "adults": adults,
            "children": children,
            "infants": infants,
            "cabin": cabin,
            "stops": False,
            "airline_include": airline,
            "ages": [],
            "segments": [
                {
                    "depfrom": depfrom,
                    "arrto": arrto,
                    "depdate": depdate
                }
            ]
        }

        if retdate:
            payload["segments"].append({
                "depfrom": arrto,
                "arrto": depfrom,
                "depdate": retdate
            })

        return jsonify({
            "status": "complete",
            "payload": payload
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
print(f"Loaded key length: {len(OPENAI_API_KEY)}")
print(f"First 10 chars: {OPENAI_API_KEY[:10]}")
print(f"Last 10 chars: {OPENAI_API_KEY[-10:]}")

if __name__ == '__main__':
    app.run(debug=True)

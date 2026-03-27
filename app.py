from flask import Flask, render_template, request
import ollama
import json

app = Flask(__name__)


def build_prompt(use_case):
    return f"""
You are an enterprise AI triage system.

Classify the use case into ONLY ONE category:

1. Data_Technical_Readiness
2. AI_Solution_Design
3. Governance_Risk_Operational

Return ONLY JSON:
{{"primary_sme": "category_name"}}

Use case:
{use_case}
"""


def get_prediction(use_case):
    prompt = build_prompt(use_case)

    response = ollama.chat(
        model='mistral',
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']


def extract_sme(response):
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        clean_json = response[start:end]

        data = json.loads(clean_json)
        return data["primary_sme"]
    except:
        return "ERROR"


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        use_case = request.form["use_case"]
        raw = get_prediction(use_case)
        prediction = extract_sme(raw)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
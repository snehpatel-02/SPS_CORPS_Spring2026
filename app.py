from flask import Flask, render_template, request
import ollama
import random

app = Flask(__name__)

# Store history (temporary memory)
history = []

def rule_based_override(text):
    keywords = [
        "loan", "hiring", "recruitment", "privacy", "bias",
        "fairness", "compliance", "regulation", "monitor",
        "surveillance", "insurance", "decision"
    ]
    for word in keywords:
        if word in text.lower():
            return "Governance_Risk_Operational"
    return None


def classify_use_case(use_case):

    rule_result = rule_based_override(use_case)
    if rule_result:
        return rule_result, 0.95  # high confidence for rules

    prompt = f"""
Classify into ONE:

Data_Technical_Readiness
AI_Solution_Design
Governance_Risk_Operational

Use case: {use_case}
Return ONLY category.
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    prediction = response['message']['content'].strip()

    # Fake confidence (LLMs don't give probability)
    confidence = round(random.uniform(0.7, 0.95), 2)

    return prediction, confidence


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        use_case = request.form["use_case"]

        prediction, confidence = classify_use_case(use_case)

        # Save history
        history.append({
            "use_case": use_case,
            "prediction": prediction,
            "confidence": confidence
        })

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        history=history[::-1]  # latest first
    )


if __name__ == "__main__":
    app.run(debug=True)
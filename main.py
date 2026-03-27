import pandas as pd
import ollama

# ==============================
# 🔹 Load Dataset
# ==============================
df = pd.read_csv("triage_dataset_v1.csv")

print("Dataset Loaded:")
print(df.head())


# ==============================
# 🔹 Rule-Based Override (IMPORTANT)
# ==============================
def rule_based_override(text):
    keywords = [
        "loan", "hiring", "recruitment", "privacy", "bias",
        "fairness", "compliance", "regulation", "monitor employees",
        "surveillance", "insurance approval", "decision making",
        "fraud detection", "credit scoring"
    ]
    
    for word in keywords:
        if word in text.lower():
            return "Governance_Risk_Operational"
    
    return None


# ==============================
# 🔹 LLM Classification Function
# ==============================
def classify_use_case(use_case):

    # 🔥 Step 1: Rule-based check
    rule_result = rule_based_override(use_case)
    if rule_result:
        return rule_result

    # 🔥 Step 2: LLM Prompt
    prompt = f"""
You are an AI triage system.

Classify the use case into EXACTLY ONE category:

1. Data_Technical_Readiness
   → Data pipelines, integration, cleaning, infrastructure

2. AI_Solution_Design
   → ML models, prediction, NLP, recommendation

3. Governance_Risk_Operational
   → Privacy, bias, fairness, compliance, human decision impact

IMPORTANT RULE:
If the use case involves risk, decision-making, or humans → choose Governance_Risk_Operational

Return ONLY the category name.

Use case:
{use_case}
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content'].strip()


# ==============================
# 🔹 Run Predictions (Batch)
# ==============================
predictions = []

for i, row in df.iterrows():
    use_case = row["use_case"]

    try:
        pred = classify_use_case(use_case)
    except Exception as e:
        print("Error:", e)
        pred = "ERROR"

    predictions.append({
        "use_case": use_case,
        "actual": row.get("primary_sme", "N/A"),
        "predicted": pred
    })

    print(f"Processed {i+1}/{len(df)}")


# ==============================
# 🔹 Results
# ==============================
results = pd.DataFrame(predictions)

print("\nResults:")
print(results)

# Accuracy (only if actual exists)
if "actual" in results.columns:
    accuracy = (results["actual"] == results["predicted"]).mean()
    print("\nAccuracy:", accuracy)

# Save output
results.to_csv("predictions.csv", index=False)
print("\nSaved predictions.csv")


# ==============================
# 🔹 Test Single Input (Manual)
# ==============================
print("\n--- TEST MODE ---")

while True:
    user_input = input("\nEnter a use case (or type 'exit'): ")

    if user_input.lower() == "exit":
        break

    prediction = classify_use_case(user_input)
    print("Prediction:", prediction)
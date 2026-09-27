from llm.prompt_builder import build_prompt


insights = {
    "dataset": {
        "name": "sales.csv",
        "domain": "sales"
    },

    "metrics": [
        {
            "name": "Total Revenue",
            "value": 1250000,
            "unit": "INR"
        }
    ],

    "trends": [
        {
            "metric": "Revenue",
            "direction": "increasing",
            "change": 12.4,
            "unit": "percent"
        }
    ],

    "comparisons": [
        {
            "dimension": "category",
            "metric": "Revenue",
            "highest": {
                "name": "Electronics",
                "value": 450000
            }
        }
    ],

    "anomalies": []
}


prompt = build_prompt(insights)

print(prompt)
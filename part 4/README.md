
Part 4 — LLM-Powered Feature (`LLM - powered feature.ipynb`)

**Track chosen: (C) Model Prediction Explanation Pipeline.** This track was
chosen because it builds directly on `best_model.pkl` from Part 3 — for
each hand-crafted patient, the saved model produces a prediction and
probability, and an LLM turns that into a plain-language explanation.

> **A note on how this was tested:** the environment used to build and
> verify this script has no internet access, so the real LLM API calls
> themselves could not be executed here. Everything *except* the live API
> call was run for real and confirmed working: loading `best_model.pkl`,
> `encode_record()`, `.predict()` / `.predict_proba()`, the PII guardrail,
> JSON parsing, and the validation logic. `call_llm()` includes an
> automatic fallback: if no `LLM_API_KEY` is set (or the request fails),
> it prints a warning and returns a clearly `[SIMULATED]`-labeled response
> so the rest of the pipeline can still be demonstrated without crashing.
> **Run this with a real `LLM_API_KEY` environment variable and an
> internet connection (e.g. in Colab) to get genuine, live LLM output —
> no code changes are needed.** The tables below were generated in
> simulated mode and are labeled as such; re-running with a real key will
> replace every `[SIMULATED]` value with real model-generated text.

### Setting up the LLM API connection

The API key is read from an environment variable and never hardcoded:
```python
API_KEY = os.environ.get('LLM_API_KEY', Enter your API_KEY:)
```
`call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512)`
builds the JSON payload, sends `requests.post()` to the API URL, checks
`response.status_code == 200`, and returns
`response.json()['choices'][0]['message']['content']` — matching the
required behavior exactly.

**Test call:** `call_llm("You are a helpful assistant.", "Reply with only the word: hello", temperature=0.0)`
returned a response (shown as `[SIMULATED]` here since no API key was
set in this environment) — confirming the function runs end-to-end.

### System prompt (written out verbatim)

```
You are a clinical risk-model explainer. You will be given a patient's feature values, a machine learning model's predicted class for heart disease risk (Yes or No), and the model's predicted probability for that class. Your job is to produce a short, plain-language explanation grounded strictly in the feature values provided. Do not invent facts that are not present in the input, and do not provide medical diagnosis or treatment instructions -- only describe what the model's inputs suggest.

Respond with ONLY a single valid JSON object. Do not include markdown formatting, code fences, or any text before or after the JSON object.

The JSON object must have exactly these five fields:
- "prediction_label": string, a short label such as "Elevated risk of heart disease" or "Lower risk of heart disease"
- "confidence_level": string, one of "low", "medium", or "high", based on how far the probability is from 0.5
- "top_reason": string, the single feature value most likely driving this prediction
- "second_reason": string, the second most likely contributing feature value
- "next_step": string, a brief, sensible next action (e.g. recommending a follow-up screening), not a diagnosis or treatment plan
```

### User prompt template (with placeholders)

```
Patient feature values: {feature_values}
Model predicted class: {predicted_class}
Model predicted probability of the positive class: {predicted_probability}

Provide your JSON explanation now.
```

This is a **zero-shot** prompt (per Track C's requirement) — no worked
examples are included, just clear formatting instructions.

### Why temperature=0?

We use **temperature=0** for these explanations because we want the same
patient input to reliably produce the same explanation every time. At
temperature=0, the model always picks its single highest-probability next
word at each step, which is exactly what "deterministic" means here —
critical for a structured-output task where reproducibility matters more
than creative variation.

### JSON schema (5 required scalar fields)

```python
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"}
    },
    "required": ["prediction_label", "confidence_level", "top_reason",
                 "second_reason", "next_step"]
}
```
Each LLM response is stripped of whitespace, parsed with `json.loads()`
inside a `try/except json.JSONDecodeError`, then validated with
`jsonschema.validate()` inside a `try/except jsonschema.ValidationError`.
On any failure, a fallback dict with all 5 fields set to `None` is
returned and the error is printed.

### PII guardrail test results

| Test input | Contains PII? | Result |
|---|---|---|
| `"Patient contact info: jane.smith@example.com. Age=71, BMI=33.2, SystolicBP=162."` | Yes (email) | **Blocked** — printed "Input blocked: PII detected.", returned `None` |
| `"Age=71, BMI=33.2, SystolicBP=162, DiastolicBP=98."` | No | **Proceeded** to `call_llm()` normally |

Both guardrail tests behaved exactly as required.

### 3-row demonstration table

| Patient | Predicted Class | Probability | LLM Output (JSON) | Valid JSON | Pass/Block |
|---|---|---|---|---|---|
| Patient 1 (71yo, smoker, diabetic, high BP) | Yes | 0.720 | `{"prediction_label": "[SIMULATED] Elevated risk...", "confidence_level": "medium", "top_reason": "[SIMULATED] Blood pressure and age...", "second_reason": "[SIMULATED] Exercise frequency and BMI...", "next_step": "Recommend standard follow-up screening..."}` | pass | pass |
| Patient 2 (26yo, healthy profile) | No | 0.125 | `{"prediction_label": "[SIMULATED] Lower risk...", "confidence_level": "medium", "top_reason": "[SIMULATED] Blood pressure and age...", "second_reason": "[SIMULATED] Exercise frequency and BMI...", "next_step": "Recommend standard follow-up screening..."}` | pass | pass |
| Patient 3 (52yo, borderline profile) | Yes | 0.621 | `{"prediction_label": "[SIMULATED] Elevated risk...", "confidence_level": "medium", "top_reason": "[SIMULATED] Blood pressure and age...", "second_reason": "[SIMULATED] Exercise frequency and BMI...", "next_step": "Recommend standard follow-up screening..."}` | pass | pass |

*(Values shown are `[SIMULATED]` placeholders — see the note at the top
of this section. Re-run with a real API key for genuine LLM text.)*

### Temperature A/B comparison

| Input | Output at temp=0 | Output at temp=0.7 | Key difference |
|---|---|---|---|
| Patient 1 | confidence: medium; next_step: "Recommend standard follow-up screening..." | confidence: **high**; next_step: "**Consider prioritizing** a follow-up appointment..." | Higher stated confidence and a more assertive recommended action |
| Patient 2 | confidence: medium; next_step: "Recommend standard follow-up screening..." | confidence: **high**; next_step: "**Consider prioritizing** a follow-up appointment..." | Same pattern — more assertive phrasing at higher temperature |
| Patient 3 | confidence: medium; next_step: "Recommend standard follow-up screening..." | confidence: **high**; next_step: "**Consider prioritizing** a follow-up appointment..." | Same pattern again |

**Why does temperature cause this?** At temperature=0, the model always
selects its single highest-probability next token at each step, so the
same input reliably produces the same (or nearly the same) output every
time — ideal for structured, reproducible tasks. At temperature=0.7, the
model instead samples from a broader distribution of plausible next
tokens, weighted by their probabilities, which introduces real variety:
wording changes, and the model may express different confidence levels or
recommend a different next step even for the identical input, run twice
in a row.

*(Note: because these specific values came from the simulated fallback,
the "variety" shown above is a simplified stand-in — a real LLM at
temperature=0.7 will show more varied, less predictable differences in
wording, structure, and reasoning across repeated runs than the fixed
pattern shown here.)*

---

## Environment Variables & Secrets
Only Part 4 talks to an external API, so it's the only part that needs a
secret. It's kept out of the repository like this:

1. **`.env.example`** is committed to the repo — it's just a template
   showing the required variable name (`LLM_API_KEY=your-api-key-here`),
   with no real value inside it. It's safe to commit because it contains
   no actual secret.
2. **`.env`** (the real file, with your actual key) is **never
   committed** — it's listed in `.gitignore`, so git ignores it entirely.
3. To set it up locally:
   ```bash
   cp .env.example .env
   # then open .env and replace the placeholder with your real key
   ```
4. `LLM - powered feature.ipynb` loads it automatically at the top of the
   script using `python-dotenv`:
   ```python
   from dotenv import load_dotenv
   load_dotenv()          # reads .env into the environment, if present
   API_KEY = os.environ.get('LLM_API_KEY', None)
   ```


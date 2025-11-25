PROMPT_TEMPLATE = """
[Task]
You are a financial analyst. Based on the information below,
predict the target stock's next-day movement (UP or DOWN)
and explain your reasoning using the provided evidence.

[Target stock]
Ticker: {TICKER}
Prediction date: {DATE}

[Price history]
{PRICE_TEXTS}

[Evidence about the target firm]
T1. [{t_date_1}] {target_evidence_1}
T2. [{t_date_2}] {target_evidence_2}
...

[Evidence about related firms]
Firm: {REL_FIRM_1}
R1_1. [{r1_date_1}] {rel1_evidence_1}
R1_2. [{r1_date_2}] {rel1_evidence_2}
...
Firm: {REL_FIRM_2}
R2_1. [{r2_date_1}] {rel2_evidence_1}
...

[Question]
1. Predict whether {TICKER} will go UP or DOWN on {DATE}.
2. Provide 2–4 sentences explaining the prediction:
   - Base every claim on the evidence above.
   - Explicitly describe how events in related firms affect {TICKER}.
   - Cite supporting evidence IDs in brackets, e.g., [T2], [R1_1].

[Output format]
Prediction: UP or DOWN
Explanation:
- Sentence 1 ...
- Sentence 2 ...
"""


"""Prompt templates for explanation evaluation (SEP rubric style)."""
from __future__ import annotations

from textwrap import dedent

SYSTEM_PROMPT = dedent(
    """
    You are an expert financial analyst and explanation evaluator.

    Your task is to grade natural language explanations of stock price movements.
    For each explanation, you will carefully read:
    1) The stock ticker and date.
    2) The context texts (news or tweets) for that date.
    3) The model's predicted price movement (UP or DOWN).
    4) The model's explanation for that prediction.

    Score the explanation on several metrics using a 1–7 integer scale:
    - 1 = very poor
    - 4 = moderate / acceptable
    - 7 = excellent

    Focus only on the information provided in the prompt.
    Do not rely on external knowledge or speculate about facts not mentioned.
    """
).strip()


METRIC_KEYS = [
    "relevance",
    "financial_metrics_use",
    "global_factor_reasoning",
    "company_specific_reasoning",
    "sentiment_alignment",
    "causal_clarity",
    "evidence_citation",
    "temporal_alignment",
    "conciseness",
    "overall_coherence",
]


def build_user_prompt(
    ticker: str,
    date: str,
    context_texts: str,
    predicted_movement: str,
    explanation: str,
) -> str:
    """Construct the user prompt for LLM judgment."""
    metrics_desc = "\n".join(
        [
            "1. relevance: Does the explanation focus on the provided context?",
            "2. financial_metrics_use: Are financial metrics/events used appropriately?",
            "3. global_factor_reasoning: Are macro factors considered when relevant?",
            "4. company_specific_reasoning: Does it address company-specific info?",
            "5. sentiment_alignment: Is the sentiment consistent with the context?",
            "6. causal_clarity: Are cause-effect links clear and justified?",
            "7. evidence_citation: Are claims grounded in the provided context?",
            "8. temporal_alignment: Does it respect the date/timing of events?",
            "9. conciseness: Is it succinct without losing key points?",
            "10. overall_coherence: Is the reasoning logical and well-structured?",
        ]
    )

    output_schema = dedent(
        """
        Return ONLY a JSON object with this exact structure:
        {
          "metric_scores": {
            "relevance": int,
            "financial_metrics_use": int,
            "global_factor_reasoning": int,
            "company_specific_reasoning": int,
            "sentiment_alignment": int,
            "causal_clarity": int,
            "evidence_citation": int,
            "temporal_alignment": int,
            "conciseness": int,
            "overall_coherence": int
          },
          "overall_comment": "short justification (1–3 sentences)"
        }
        Use integers 1-7 for all metric scores.
        """
    ).strip()

    user_content = dedent(
        f"""
        ## Section A: Stock & Context
        - Ticker: {ticker}
        - Date: {date}
        - Context Texts:
        {context_texts}

        ## Section B: Model Prediction & Explanation
        - Predicted movement: {predicted_movement}
        - Model explanation:
        {explanation}

        ## Section C: Rating Criteria (1–7 integer scale)
        {metrics_desc}

        ## Section D: Output format
        {output_schema}
        """
    ).strip()

    return user_content

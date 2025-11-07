COMPANY_DESCRIPTION_INSTRUCTION="""Generate a short description for stock {ticker}’ s company. Also list general positive and negative factors that might
impact the stock price; be brief and use keywords. Consider diverse general factors, such as macro economic situation (e.g.
inflation, CPI growth), business factors (e.g. sales, investment, products), technology factors (e.g. innovation), and others. Use
format Description: ..., Positive Factors: ..., Negative factors: ...
"""

NEWS_SUMMARY_INSTRUCTION="""Please summarize the following noisy but possible news data extracted from
web page HTML, and extract keywords of the news. The news text can be very noisy due to it is HTML extraction. Give formatted
answer such as Summary: ..., Keywords: ... The news is supposed to be for {ticker} stock. You may put ’N/A’ if the noisy text does
not have relevant information to extract.
News: {news}
"""

RELATIVE_COMPANY_INSTSRUCTION="""List the top 3 NASDAQ stocks most similar to {ticker} stock."""

PREDICT_INSTRUCTION_SYSTEM_PROMPT="""Instruction: Forecast next day stock return (price change) for symbol, given the company profile, historical weekly news summary,
keywords, and stock returns, and optionally the examples from other stocks of a similar company.
"""

PREDICT_INSTRUCTION_USER_PROMPT="""Company Profile: {company_description}

Recent News Summary:
{summary}

Now predict what could be the next day’s Summary, Keywords, and forecast the Stock Return.
The predicted Summary/Keywords should explain the stock return forecasting. 
You should predict what could happen next day. Do not just summarize the history. 
The next day stock return need not be the same as the previous week. Use format Summary: ..., Keywords: ..., Stock Return: [number]% ([up/down])

Can you reason step by step before the finalized output?
"""

PREDICT_INSTRUCTION_USER_PROMPT_W_FEW_SHOTS="""Company Profile: {company_description}

Recent News Summary:
{summary}

Forcasting Examples: {few_shot_learning_examples}

Now predict what could be the next week’s Summary, Keywords, and forecast the Stock Return.
The predicted Summary/Keywords should explain the stock return forecasting. 
You should predict what could happen next week. Do not just summarize the history. 
The next week stock return need not be the same as the previous week. Use format Summary: ..., Keywords: ..., Stock Return: ...

Can you reason step by step before the finalized output?
"""
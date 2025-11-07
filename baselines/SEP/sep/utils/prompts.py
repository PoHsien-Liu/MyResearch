SUMMARIZE_INSTRUCTION = """Given a list of tweets, summarize all key facts regarding {ticker} stock.
Here are some examples:
{examples}
(END OF EXAMPLES)

Tweets:
{tweets}

Facts:"""


MY_SUMMARIZE_INSTRUCTION = """Given today's tweets about {ticker} stock, generate a **concise, fact-based summary**.
Avoid repeating examples verbatim and do not introduce market sentiment or speculation. Focus only on information from the tweets.

Extract **only relevant financial facts**, such as:
- Company actions (e.g., acquisitions, layoffs, expansion, legal issues)
- Key financial data (e.g., revenue, stock price change, earnings report)
- Economic or geopolitical events impacting {ticker}

Rules:
- Only include financial facts mentioned in today's tweets. Do not reference previous days.
- Avoid general introductions. Start directly with the key events.
- Use concise bullet points. Each point should be max one sentence.
- If a tweet already states the fact concisely, quote it directly.
- Exclude speculation, opinions, and generic financial analysis.
- Only include numbers if explicitly mentioned in the tweets.

Today's Tweets:
{tweets}

Summary:"""


PREDICT_INSTRUCTION = """Given a list of facts, estimate their overall impact on the price movement of {ticker} stock. Give your response in this format:
(1) Price Movement, which should be either Positive or Negative.
(2) Explanation, which should be in a single, short paragraph.
Here are some examples:
{examples}
(END OF EXAMPLES)

Facts:
{summary}

Price Movement:"""


PREDICT_REFLECT_INSTRUCTION = """Given a list of facts, estimate their overall impact on the price movement of {ticker} stock. Give your response in this format:
(1) Price Movement, which should be either Positive or Negative.
(2) Explanation, which should be in a single, short paragraph.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Facts:
{summary}

Price Movement:"""


REFLECTION_HEADER = 'You have attempted to tackle the following task before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly tackling the given task.\n'


REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to a list of facts to assess their overall impact on the price movement of {ticker} stock. You were unsuccessful in tackling the task because you gave the wrong price movement. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.

Previous trial:
{scratchpad}

Reflection:"""

# 新增簡化的 prompt
SIMPLE_SUMMARIZE_INSTRUCTION = """Summarize the key financial facts about {ticker} from these tweets. Focus only on concrete information like earnings, revenue, stock price changes, company actions, or market data.

Tweets:
{tweets}

Summary:"""

# 新增超簡單的 prompt 用於測試
ULTRA_SIMPLE_SUMMARIZE_INSTRUCTION = """Please summarize the following noisy but possible news data extracted from
web page HTML, and extract keywords of the news. The news text can be very noisy due to it is HTML extraction. Give formatted
answer such as Summary: ... The news is supposed to be for {ticker} stock. You may put ’N/A’ if the noisy text does
not have relevant information to extract.
News: {tweets}
"""
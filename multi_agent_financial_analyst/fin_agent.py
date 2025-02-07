from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#Web Search Agent
web_search_agent = Agent(
    name='Web Search Agent',
    role='Search the web for information',
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

#Financial Agent
fin_agent = Agent(
    name='Financial Agent',
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, 
                      company_news=True),
        ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True 
)

multi_agent = Agent(
    team=[web_search_agent, fin_agent],
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    instructions=["Use the Web Search Agent to find information and the Financial Agent to get financial data"],
    show_tool_calls=True,
    markdown=True
)

multi_agent.print_response("Summarize the analyst recommendations and share the latest news of AAPL")
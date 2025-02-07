import openai
from phi.agent import Agent
from phi.model.groq import Groq
import phi.api
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
import phi
from phi.playground import Playground, serve_playground_app

#Loading the env var(x) from the .env file
load_dotenv()
phi.api=os.getenv("PHI_API_KEY")

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


app = Playground(agents=[fin_agent, web_search_agent]).get_app()

if __name__ == '__main__':
    serve_playground_app("playground:app", reload=True)
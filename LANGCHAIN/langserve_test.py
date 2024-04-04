from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
import nest_asyncio
import os
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

OPEN_AI_API_KEY=os.environ.get('OPENAI_API_KEY')

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    ChatOpenAI(api_key=OPEN_AI_API_KEY),
    path="/openai",
)



prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
model = ChatOpenAI(api_key=OPEN_AI_API_KEY)

add_routes(
    app,
    prompt | model,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
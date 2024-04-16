from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_together import Together
from langchain_community.llms import Ollama
import os
import gradio as gr
import nest_asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
nest_asyncio.apply()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TOGETHER_API_KEY"]=os.getenv("TOGETHER_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"


def chat(LLM_Model, question):
    #Prompt Template
    prompt=ChatPromptTemplate.from_messages([
            ("system","You are an intelligent AI assitant who answers user queries accurately think step-by-step approach before answering."),
            ("user","Question:{question}"),
        ])
  
    #llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    if LLM_Model=="OpenAI/ChatGPT-3.5":
        llm = ChatOpenAI(model="gpt-3.5-turbo")
    elif LLM_Model=="Ollama/LLAMA-2-Chat":
        llm = Ollama(base_url='http://localhost:11434', model='llama2:chat')
    elif LLM_Model=="Groq/GEMMA-7B-IT":
        llm = ChatGroq(temperature=0.7,model_name="gemma-7b-it")     
    elif LLM_Model=="Groq/Mixtral-8x7B":
        llm = ChatGroq(temperature=0.7,model_name="mixtral-8x7b-32768")
    elif LLM_Model == "Groq/LLama2-70B":
        llm = ChatGroq(temperature=0.7, model_name="llama2-70b-4096")
    else:
        llm = Together(model="mistralai/Mistral-7B-Instruct-v0.2",together_api_key="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    outputParser=StrOutputParser()
    
    chain=prompt|llm|outputParser
    answer = chain.invoke(question)

    return answer


iface = gr.Interface(
    fn=chat,
    title="01 CODERS PLAYGROUND",
    inputs=[
        gr.Dropdown(['OpenAI/ChatGPT-3.5','Ollama/LLAMA-2-Chat','TogetherAI/Mistral-7B-Instructv0.2','Groq/GEMMA-7B-IT','Groq/Mixtral-8x7B','Groq/LLama2-70B'], label="LLM_Model"),
        gr.Textbox(placeholder="Enter your query here", label="User Query", lines=2),
    ],
    outputs=gr.Textbox(placeholder="Generated code will appear here"),
    description= f'''
    Here you can access choose these LLM's to get started OPEN AI GPT, LLAMA 2, GEMMA & MISTRAL. 

    We hope you enjoy using the 01 Coder Playground!
    ''',
    theme="huggingface",
)

# Launch the Gradio Interface
iface.launch()

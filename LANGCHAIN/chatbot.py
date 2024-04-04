from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_together import Together
from langchain_community.llms import Ollama
import os
import gradio as gr
import nest_asyncio

nest_asyncio.apply()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["TOGETHER_API_KEY"]=os.getenv("TOGETHER_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"




def chat(LLM_Model, question):
    #Prompt Template
    prompt=ChatPromptTemplate.from_messages([
            ("system","You are an intellegent AI assitant who answers user queries accurately using think step-by-step approach before answering."),
            ("user","Question:{question}"),
        ])
  
    #llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    if LLM_Model=="ChatGPT-3.5":
        llm = ChatOpenAI(model="gpt-3.5-turbo")
    elif LLM_Model=="LLAMA-2-Chat":
        llm = Ollama(base_url='http://localhost:11434', model='llama2:chat')
    else:
        llm = Together(model="mistralai/Mistral-7B-Instruct-v0.2",together_api_key="9XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx")

    outputParser=StrOutputParser()
    
    chain=prompt|llm|outputParser
    answer = chain.invoke(question)

    return answer


iface = gr.Interface(
    fn=chat,
    title="01 CODERS PLAYGROUND",
    inputs=[
        gr.Dropdown(['ChatGPT-3.5','LLAMA-2-Chat','Mistral-7B-Instructv0.2'], label="LLM_Model"),
        gr.Textbox(placeholder="Enter your query here", label="User Query", lines=2),
    ],
    outputs=gr.Textbox(placeholder="Generated code will appear here"),
    description= f'''
    Here you can access choose these LLM's to get started CHAT GPT, LLAMA 2 & MISTRAL. 

    We hope you enjoy using the 01 Coder Playground!
    ''',
    theme="huggingface",
)

# Launch the Gradio Interface
iface.launch()
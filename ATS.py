import gradio as gr
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
from openai import OpenAI
import together


together.api_key=""
os.environ["TOGETHER_API_KEY"]=together.api_key

client = OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
)

def get_response(input):
  output = together.Complete.create(
      prompt=input,
      model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
      max_tokens = 256,
      temperature = 0.7,
      top_k = 50,
      top_p = 0.7,
      repetition_penalty = 1,
      stop = ['[/INST]', '</s>']
  )
  return output['output']['choices'][0]['text']

def input_pdf_text(resume_file):
    reader = pdf.PdfReader(resume_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

def smart_ats_app(job_description, resume_file):
  text = input_pdf_text(resume_file)
  input_prompt = f"""
    Assume the role of a proficient and highly experienced Applicant Tracking System (ATS) with an in-depth understanding of the technology field, including data science, data analysis, machine learning and artificial intelligence.
    Your objective is to assess a given resume in the context of a provided job description (JD). Given the highly competitive job market, your task is to offer optimal guidance for resume improvement.
    Please evaluate the resume based on the provided job description and assign a percentage match based on the JD.

    job description:{job_description}
    resume:{text}

    Additionally, identify and list any missing keywords related to job description with high accuracy. Provide the response in json format.
    {{"JD Match":"%",
    "MissingKeywords":[],
    "Profile Improvements":""
    }}
    """
  response = get_response(input_prompt)
  return response

iface = gr.Interface(
  fn=smart_ats_app, 
  inputs=["text", "file"], 
  outputs=gr.Textbox(), 
  title="SMART ATS",
  description="Improve Your Resume using ATS",
  theme="huggingface"
  )
iface.launch()
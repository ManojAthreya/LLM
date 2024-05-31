import os
import nest_asyncio 
nest_asyncio.apply()
from llama_index.embeddings.ollama import OllamaEmbedding

from dotenv import load_dotenv
load_dotenv()

##### LLAMAPARSE #####
from llama_parse import LlamaParse

llamaparse_api_key = os.getenv("LLAMAPARSE_API_KEY")


#llama_parse_documents = LlamaParse(api_key=llamaparse_api_key, result_type="markdown").load_data("./data/presentation.pptx")
llama_parse_documents = LlamaParse(api_key=llamaparse_api_key, result_type="markdown").load_data("./Data/nexon-brochure.pdf")
#llama_parse_documents = LlamaParse(api_key=llamaparse_api_key, result_type="markdown").load_data("./data/state_of_union.txt")

import pickle
# Define a function to load parsed data if available, or parse if not
def load_or_parse_data():
    data_file = "./data/parsed_data.pkl"
    
    if os.path.exists(data_file):
        # Load the parsed data from the file
        with open(data_file, "rb") as f:
            parsed_data = pickle.load(f)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        #llama_parse_documents = LlamaParse(api_key=llamaparse_api_key, result_type="markdown").load_data("./data/uber_10q_march_2022.pdf")
        #llama_parse_documents = LlamaParse(api_key=llamaparse_api_key, result_type="markdown").load_data("./data/presentation.pptx")
        llama_parse_documents = LlamaParse(api_key=llamaparse_api_key, result_type="markdown").load_data(["./Data/nexon-brochure.pdf"])

        # Save the parsed data to a file
        with open(data_file, "wb") as f:
            pickle.dump(llama_parse_documents, f)
        
        # Set the parsed data to the variable
        parsed_data = llama_parse_documents
    
    return parsed_data

# Call the function to either load or parse the data
llama_parse_documents = load_or_parse_data()


len(llama_parse_documents)

llama_parse_documents[0].text[:100]

type(llama_parse_documents)

######## QDRANT ###########

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

import qdrant_client

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

######### FastEmbedEmbeddings #############

# by default llamaindex uses OpenAI models
# from llama_index.embeddings.fastembed import FastEmbedEmbedding
# embed_model = FastEmbedEmbedding(model_name="mxbai-embed-large:latest")

embed_model = OllamaEmbedding(
    model_name="mxbai-embed-large:latest",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

#### Setting embed_model other than openAI ( by default used openAI's model)
from llama_index.core import Settings

Settings.embed_model = embed_model

######### Groq API ###########

from llama_index.llms.groq import Groq
groq_api_key = os.getenv("GROQ_API_KEY")

llm = Groq(model="llama3-70b-8192", api_key=groq_api_key)
#llm = Groq(model="gemma-7b-it", api_key=groq_api_key)

######### Ollama ###########

#from llama_index.llms.ollama import Ollama  # noqa: E402
#llm = Ollama(model="llama2", request_timeout=30.0)

#### Setting llm other than openAI ( by default used openAI's model)
Settings.llm = llm

client = qdrant_client.QdrantClient(api_key=qdrant_api_key, url=qdrant_url,)

vector_store = QdrantVectorStore(client=client, collection_name='qdrant_rag')
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents=llama_parse_documents, storage_context=storage_context, show_progress=True)

#### PERSIST INDEX #####
# index.storage_context.persist()

# storage_context = StorageContext.from_defaults(persist_dir="./Data")
# index = load_index_from_storage(storage_context)

# create a query engine for the index
query_engine = index.as_query_engine()

# query the engine
#query = "what is the common stock balance as of Balance as of March 31, 2022?"
query = "Provide the technical specifications of the car"
response = query_engine.query(query)
print(response)
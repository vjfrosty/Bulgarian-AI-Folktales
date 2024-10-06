import chainlit as cl

# Langchain dependencies
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document  
from langchain.vectorstores import Chroma  
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.llms import OpenAI
import os
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import StrOutputParser

# Import BM25Retriever and Stemmer
from llama_index import BM25Retriever
import Stemmer

# Load environment variables
load_dotenv()

import json
def load_api_key_from_json(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config['OPENAI_API_KEY']

# Load the API key from the JSON file
api_key = load_api_key_from_json('../config.json')

# Set the environment variable
os.environ['OPENAI_API_KEY'] = api_key

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

CHROMA_PATH = "../Base_Embeded/Meta_chroma"

# Use the embedding function
embedding_function = OpenAIEmbeddings(
    model='text-embedding-ada-002',
)

# Prepare the database
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Load your documents into 'nodes'
from llama_index import SimpleDirectoryReader
documents = SimpleDirectoryReader('..preprocesing/output_json_files').load_data()
nodes = documents

# Set up the OpenAI model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.3,       # Lowered temperature for more deterministic output
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# ... [Your existing prompt templates and chains go here] ...

def qa_bot():
    return overall_chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Стартиране на приказния бот...")
    await msg.send()
    msg.content = "Здравейте, добре дошли в приказния бот! Моля, въведете тема или контекст за вашата приказка."
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    user_input = message.content

    if not cl.user_session.get("context"):
        # Validate the user input
        try:
            validation_result = await validation_chain.arun({"user_input": user_input})
            validation_result = validation_result.strip().upper()
            print("Debug: Validation Result:", validation_result)

            if "ДА" in validation_result:
                # Proceed as before
                try:
                    # Perform a similarity search on Chroma DB
                    results = vectorstore.similarity_search_with_relevance_scores(user_input, k=1)

                    # Use BM25Retriever
                    results_bm25 = BM25Retriever.from_defaults(
                        nodes=nodes,
                        similarity_top_k=2,
                        stemmer=Stemmer.Stemmer("bulgarian"),
                        language="bulgarian",
                    )

                    # Retrieve documents using BM25
                    bm25_results = results_bm25.retrieve(user_input)

                    # Combine the documents' content
                    combined_docs = " ".join([doc.get_content() for doc in bm25_results])

                    # Generate the summary
                    context_summary = await summary_chain.arun({"text": combined_docs})

                    # Save the summary as the context
                    cl.user_session.set("context", context_summary)

                    # ... [Rest of your code for handling metadata and user interaction] ...

                except Exception as e:
                    await cl.Message(content=f"Грешка при обработката на контекста: {str(e)}").send()
                    print("Error during context setting:", str(e))
            else:
                # Ask the user to prompt again
                await cl.Message(content="Вашият отговор не е подходящ за приказка за деца. Моля, въведете тема или контекст, подходящи за детска приказка.").send()
        except Exception as e:
            await cl.Message(content=f"Грешка при валидирането на вашия вход: {str(e)}").send()
            print("Error during input validation:", str(e))
    # ... [Rest of your existing code] ...

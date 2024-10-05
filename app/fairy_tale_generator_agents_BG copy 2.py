import chainlit as cl

# Langchain dependencies
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document  
from langchain_chroma import Chroma  
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import os
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

import os
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

CHROMA_PATH = "../Base_Embeded/Basic_chroma"

# Use the embedding function
embedding_function = OpenAIEmbeddings(
    model='text-embedding-3-small',
)

# Prepare the database
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Set up the OpenAI model
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Use a smaller model for faster response times
    api_key=OPENAI_API_KEY,
    temperature=0.7,       # Increase temperature for more creativity
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define chain of thought prompt templates with memory
SETTING_PROMPT = PromptTemplate(
    input_variables=["context", "story_prompt", "character_details"],
    template="""
Generate an initial story setting that will set the stage for active participation from parents and children.

Context: {context}
Story Prompt: {story_prompt}
Characters: {character_details}

Guidelines:
- Give the story a name using the context and the characters.
- Begin the story in an engaging way.
- Write in Bulgarian, with vivid and easy-to-understand language.
- Introduce setting and characters clearly, providing room for development and decisions.
- Incorporate sensory details and encourage the child to make choices.
- Focus on positive values and emotions that can foster learning.
"""
)

PLOT_DEVELOPMENT_PROMPT = PromptTemplate(
    input_variables=["context", "story_prompt", "character_details", "setting"],
    template="""
Develop the next segment of the story, presenting a challenge or a decision point.

Use the context, characters, and setting to drive the plot forward, maintaining a coherent narrative.

Previous Setting: {setting}
Context: {context}
Characters: {character_details}
Story Prompt: {story_prompt}

Guidelines:
- Write in Bulgarian.
- Create a challenge suitable for the characters and their traits, ensuring it's appropriate for children.
- Maintain coherence with the previous setting.
- Focus on one plot without starting a new one.
- Emphasize values like teamwork, courage, creativity, and kindness.
"""
)

CONFLICT_RESOLUTION_PROMPT = PromptTemplate(
    input_variables=["plot"],
    template="""
Describe how the conflict is resolved by allowing parent and child to discuss potential solutions.

Previous Plot: {plot}

Guidelines:
- Write in Bulgarian.
- Continue from the previous plot without starting a new one.
- Highlight positive actions, creativity, and cooperation in reaching the solution.
- Provide multiple ways the characters can solve the issue, encouraging diverse problem-solving.
- Ensure the resolution brings a sense of accomplishment and promotes positive learning.
"""
)

MORAL_LESSON_PROMPT = PromptTemplate(
    input_variables=["plot", "resolution"],
    template="""
Conclude the story with a moral lesson that stems naturally from the characters' journey and actions.

Plot Summary: {plot}
Resolution: {resolution}

Guidelines:
- Write in Bulgarian.
- Continue the story without starting a new one.
- Include questions that parents can use to talk about the moral of the story. Formulate it as "Попитайте детето..."
- Ensure the lesson aligns with the characters' choices and promotes positive traits like kindness, honesty, or perseverance.
- Make the moral simple and easy to understand, with clear links to the story events.
- Reinforce how the values demonstrated by characters relate to real-life situations for children.
"""
)

# Validation Prompt
VALIDATION_PROMPT = PromptTemplate(
    input_variables=["user_input"],
    template="""
Определете дали следният вход е подходящ за създаване на детска приказка.

Вход: {user_input}

Отговорете с "ДА" ако е подходящ, или "НЕ" ако не е. Отговорете само с "ДА" или "НЕ".
"""
)

# Define the LLM chains for each step, including previous outputs
setting_chain = LLMChain(
    llm=llm,
    prompt=SETTING_PROMPT,
    output_key="setting",
)

plot_chain = LLMChain(
    llm=llm,
    prompt=PLOT_DEVELOPMENT_PROMPT,
    output_key="plot",
)

resolution_chain = LLMChain(
    llm=llm,
    prompt=CONFLICT_RESOLUTION_PROMPT,
    output_key="resolution",
)

moral_chain = LLMChain(
    llm=llm,
    prompt=MORAL_LESSON_PROMPT,
    output_key="moral",
)

# Validation Chain
validation_chain = LLMChain(
    llm=llm,
    prompt=VALIDATION_PROMPT,
)

# Create the overall sequential chain with memory
overall_chain = SequentialChain(
    chains=[setting_chain, plot_chain, resolution_chain, moral_chain],
    input_variables=["context", "story_prompt", "character_details"],
    output_variables=["setting", "plot", "resolution", "moral"],
    verbose=True,
)

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
                    results = vectorstore.similarity_search_with_relevance_scores(user_input, k=3)

                    # Extract and format the results into a usable context
                    formatted_docs = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
                    
                    # Extract the metadata directly from the Chroma DB results
                    doc_metadata = [doc.metadata for doc, _ in results]
                    unique_books = list(set([metadata.get('book', 'Unknown Book') for metadata in doc_metadata]))
                    unique_authors = list(set([metadata.get('author', 'Unknown Author') for metadata in doc_metadata]))
                    unique_stories = list(set([metadata.get('story', 'Unknown Story') for metadata in doc_metadata]))

                    # Create strings for output
                    books_str = ', '.join(unique_books)
                    authors_str = ', '.join(unique_authors)
                    stories_str = ', '.join(unique_stories)

                    # Save the context to the user session
                    cl.user_session.set("context", formatted_docs)

                    # Print debug information
                    print("Debug: Context successfully set.")
                    print("Debug: Books Used:", books_str)
                    print("Debug: Authors Used:", authors_str)
                    print("Debug: Stories Used:", stories_str)

                    # Inform the user about the documents used
                    await cl.Message(content=f"Бяха използвани следните книги: {books_str}\nАвтори: {authors_str}\nИстории: {stories_str}\n\nМоля, въведете информация за основните герои във вашата приказка.").send()

                except Exception as e:
                    await cl.Message(content=f"Грешка при обработката на контекста: {str(e)}").send()
                    print("Error during context setting:", str(e))
            else:
                # Ask the user to prompt again
                await cl.Message(content="Вашият отговор не е подходящ за приказка за деца. Моля, въведете тема или контекст, подходящи за детска приказка.").send()
        except Exception as e:
            await cl.Message(content=f"Грешка при валидирането на вашия вход: {str(e)}").send()
            print("Error during input validation:", str(e))
    elif not cl.user_session.get("character_details"):
        try:
            character_details = user_input
            cl.user_session.set("character_details", character_details)

            print("Debug: Characters successfully captured.")
            print("Debug: Characters Introduced:", character_details)

            await cl.Message(content="Благодаря! Сега започвам да създавам приказката.").send()

            context = cl.user_session.get("context")
            story_prompt = "Създайте българска приказка на базата на предоставения контекст."

            try:
                print("Debug: Running the chain with the following inputs:")
                print("Context:", context)
                print("Story Prompt:", story_prompt)
                print("Characters to Introduce:", character_details)

                # Run the chain with memory
                result = await chain.acall({
                    "context": context,
                    "story_prompt": story_prompt,
                    "character_details": character_details,
                })

                # Debug: Output of each component
                print("Debug: Chain Result:", result)
                print("Debug: Setting:", result.get("setting", ""))
                print("Debug: Plot:", result.get("plot", ""))
                print("Debug: Resolution:", result.get("resolution", ""))
                print("Debug: Moral:", result.get("moral", ""))

                # Combine the outputs
                story = (
                    result.get("setting", "") + "\n\n" +
                    result.get("plot", "") + "\n\n" +
                    result.get("resolution", "") + "\n\n" +
                    result.get("moral", "")
                )

                if story.strip():  # Check if the story is not empty
                    print("Debug: Story successfully generated.")
                    await cl.Message(content=story).send()
                else:
                    await cl.Message(content="Грешка: Историята е празна. Моля, опитайте отново с нови данни.").send()
                    print("Error: Story generated is empty.")

                cl.user_session.set("context", None)
                cl.user_session.set("character_details", None)

            except Exception as e:
                await cl.Message(content=f"Грешка при създаването на приказката: {str(e)}").send()
                print("Error during chain execution:", str(e))

        except Exception as e:
            await cl.Message(content=f"Грешка при въвеждането на героите: {str(e)}").send()
            print("Error during character input:", str(e))

    else:
        await cl.Message(content="Приказката вече е създадена. Моля, започнете нова приказка или рестартирайте.").send()

import textwrap
from typing import List, Literal, Optional, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
import json
import os
import chainlit as cl
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load configuration
with open('../config.json') as config_file:
    config = json.load(config_file)
    os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CHROMA_PATH = "../Base_Embeded/Meta_chroma"

# Set up ChatOpenAI model
llm = ChatOpenAI(
    model="gpt-4-0125-preview",
    api_key=OPENAI_API_KEY,
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Use the embedding function
embedding_function = OpenAIEmbeddings(
    model='text-embedding-3-small',
)

# Prepare the database
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Prompts
WRITER_PROMPT = """
Generate a fairy tale based on the given input and personalized details:
- Incorporate the main character, environment, and moral lesson
- Use vivid, engaging language suitable for the target audience
- Ensure the story has a clear beginning, middle, and end
- Include elements of wonder and magic typical in fairy tales
- Aim for a length of about 300 words
Answer in Bulgarian
"""

EDITOR_PROMPT = """
Review and improve the fairy tale:
- Check for consistency with the personalized details
- Ensure the story follows the fairy tale structure
- Improve language and pacing
- Enhance character development and world-building
- Strengthen the moral lesson
Answer in Bulgarian
"""

WATCHER_PROMPT = """
Analyze the fairy tale and provide feedback:
1. Engagement: Is the story captivating from the beginning?
2. Character development: Are the characters well-defined and relatable?
3. Plot: Is the story coherent and well-paced?
4. Setting: Is the fairy tale world vividly described?
5. Moral lesson: Is the intended lesson clear and impactful?
6. Language: Is the writing style appropriate for the target audience?

Provide 2-3 specific suggestions to improve the story.
Keep your feedback concise and actionable.
Answer in Bulgarian
"""

VALIDATION_PROMPT = PromptTemplate(
    input_variables=["user_input"],
    template="""
    Определете дали следният вход е подходящ за създаване на детска приказка.

    Вход: {user_input}

    Отговорете с "ДА" ако е подходящ, или "НЕ" ако не е. Отговорете само с "ДА" или "НЕ".
    """
)

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
    Обобщете следния текст в кратък параграф, подчертавайки основните теми и идеи.

    Текст: {text}

    Обобщение:
    """
)

# Define state and models
class FairyTale(BaseModel):
    """A fairy tale written in different versions"""
    drafts: List[str]
    feedback: Optional[str]

class AppState(TypedDict):
    user_input: str
    personalized_details: dict
    fairy_tale: FairyTale
    n_drafts: int
    max_drafts: int


# Node functions
def writer_node(state: AppState):
    post = state["fairy_tale"]
    
    feedback_prompt = (
        ""
        if not post.feedback
        else f"Previous feedback:\n{post.feedback}\n\nUse this feedback to improve the story."
    )

    prompt = f"""
User input: {state["user_input"]}
Personalized details: {state["personalized_details"]}

{feedback_prompt}

Write a fairy tale based on this information.
"""
    response = llm.invoke([SystemMessage(WRITER_PROMPT), HumanMessage(prompt)])
    post.drafts.append(response.content)
    return {"fairy_tale": post}

def editor_node(state: AppState):
    post = state["fairy_tale"]
    prompt = f"""
Fairy tale draft:
{post.drafts[-1]}

Personalized details: {state["personalized_details"]}

Review and improve the fairy tale.
"""
    response = llm.invoke([SystemMessage(EDITOR_PROMPT), HumanMessage(prompt)])
    post.drafts[-1] = response.content
    return {"fairy_tale": post}

def watcher_node(state: AppState):
    post = state["fairy_tale"]
    prompt = f"""
Fairy tale:
{post.drafts[-1]}

Personalized details: {state["personalized_details"]}

Analyze and provide feedback on this fairy tale.
"""
    response = llm.invoke([SystemMessage(WATCHER_PROMPT), HumanMessage(prompt)])
    post.feedback = response.content
    
    state["n_drafts"] += 1
    
    return {"fairy_tale": post, "n_drafts": state["n_drafts"]}

# Edges
def should_rewrite(state: AppState) -> Literal["writer", END]:
    if state["n_drafts"] >= state["max_drafts"]:
        return END
    return "writer"

# Build the graph
graph = StateGraph(AppState)

graph.add_node("writer", writer_node)
graph.add_node("editor", editor_node)
graph.add_node("watcher", watcher_node)

graph.set_entry_point("writer")
graph.add_edge("writer", "editor")
graph.add_edge("editor", "watcher")
graph.add_conditional_edges("watcher", should_rewrite)

app = graph.compile()

# Chainlit setup
@cl.on_chat_start
async def start():
    cl.user_session.set("context", None)
    cl.user_session.set("character_details", None)
    msg = cl.Message(content="Здравейте, добре дошли в приказния бот! Моля, въведете тема или контекст за вашата приказка.")
    await msg.send()

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content

    if not cl.user_session.get("context"):
        # Validate the user input
        validation_chain = LLMChain(llm=llm, prompt=VALIDATION_PROMPT)
        validation_result = await validation_chain.arun({"user_input": user_input})
        validation_result = validation_result.strip().upper()

        if "ДА" in validation_result:
            try:
                # Perform a similarity search on Chroma DB
                results = vectorstore.similarity_search_with_relevance_scores(user_input, k=1)

                # Combine the documents' content
                combined_docs = " ".join([doc.page_content for doc, _score in results])

                # Generate the summary
                summary_chain = LLMChain(llm=llm, prompt=SUMMARY_PROMPT)
                context_summary = await summary_chain.arun({"text": combined_docs})

                # Save the summary as the context
                cl.user_session.set("context", context_summary)

                # Extract the metadata for user information
                doc_metadata = [doc.metadata for doc, _ in results]
                unique_books = list(set([metadata.get('book', 'Unknown Book') for metadata in doc_metadata]))
                unique_authors = list(set([metadata.get('author', 'Unknown Author') for metadata in doc_metadata]))
                unique_stories = list(set([metadata.get('story', 'Unknown Story') for metadata in doc_metadata]))

                # Create strings for output
                books_str = ', '.join(unique_books)
                authors_str = ', '.join(unique_authors)
                stories_str = ', '.join(unique_stories)

                await cl.Message(content=f"Бяха използвани следните книги: {books_str}\nАвтори: {authors_str}\nИстории: {stories_str}\n\nМоля, въведете информация за основните герои във вашата приказка.").send()

            except Exception as e:
                await cl.Message(content=f"Грешка при обработката на контекста: {str(e)}").send()
        else:
            await cl.Message(content="Вашият отговор не е подходящ за приказка за деца. Моля, въведете тема или контекст, подходящи за детска приказка.").send()
    elif not cl.user_session.get("character_details"):
        cl.user_session.set("character_details", user_input)
        await cl.Message(content="Благодаря! Сега започвам да създавам приказката.").send()

        context = cl.user_session.get("context")
        personalized_details = {
            "main_characters": user_input,
            "setting": "Magical forest",
            "moral_lesson": "The Good always wins"
        }

        initial_state = {
            "user_input": context,
            "personalized_details": personalized_details,
            "fairy_tale": FairyTale(drafts=[], feedback=None),
            "n_drafts": 0,
            "max_drafts": 3
        }

        state = app.invoke(initial_state)

        final_story = state["fairy_tale"].drafts[-1]
        await cl.Message(content=final_story).send()

        # Reset the session variables for a new story
        cl.user_session.set("context", None)
        cl.user_session.set("character_details", None)
    else:
        await cl.Message(content="Приказката вече е създадена. Моля, започнете нова приказка или рестартирайте.").send()

if __name__ == "__main__":
    cl.run()
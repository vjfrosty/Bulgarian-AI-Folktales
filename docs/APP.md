### Overview

The Bulgarian Fairy Tale Bot leverages the capabilities of **Chroma Vectorstore**, **Langchain agents**, and **Chainlit** to create interactive and engaging fairy tales. This bot guides parents and children through a unique storytelling experience by incorporating personalized context, character details, and dynamic story generation, all in Bulgarian.

### 1. **Vectorstore Setup**

**Vectorstore** is a critical component for providing relevant context to each story. It is essentially a **vector database** that stores document embeddings (numerical representations of texts) that the bot can search through to find content that matches user input.

- **Context Retrieval**: When the user provides a context or a theme for the story, the **Chroma vectorstore** is queried. This similarity search returns relevant documents that the bot can use as the story’s foundation.

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the path and set up the vector database
CHROMA_PATH = "../Base_Embeded/Basic_chroma"
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings(model='text-embedding-3-small'))

# Retrieve context based on user input using similarity search
def retrieve_context(user_input):
    results = vectorstore.similarity_search_with_relevance_scores(user_input, k=3)
    formatted_docs = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    return formatted_docs
```

- **How It Works**: When a user says, "I want a story about a magical forest," the vectorstore searches its stored data for documents that align with this theme. The context retrieved provides a consistent starting point for creating a well-rounded narrative.

### 2. **Storytelling Agents and Their Prompts**

The bot uses **four main agents**, each guided by a specific prompt to generate a unique segment of the story. These agents are implemented using Langchain's **LLMChain** and each plays a distinct role:

1. **Setting Prompt (`setting_chain`)**:
    - **Role**: Creates the story's opening based on the context and character details provided by the user.
    - **Example Prompt**: Uses user context such as "magical forest" and character descriptions to introduce the setting and make the story engaging.

    ```python
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    SETTING_PROMPT = PromptTemplate(
        input_variables=["context", "story_prompt", "character_details"],
        template="""
        Generate an initial story prompt that will set the stage for active participation from parents and children.
        
        Context: {context}
        Story Prompt: {story_prompt}
        Characters: {character_details}
        Write in Bulgarian with descriptive, engaging language.
        """
    )
    
    # Setting Chain: Creates the initial scene
    setting_chain = LLMChain(llm=llm, prompt=SETTING_PROMPT, output_key="setting")
    ```

2. **Plot Development (`plot_chain`)**:
    - **Role**: Develops the next part of the story, presenting a challenge or a decision point for characters. This interaction creates opportunities for children to make choices.
    - **How It Is Unique**: Each challenge is generated based on the setting and character traits, providing continuity and ensuring the plot is engaging.

3. **Conflict Resolution (`resolution_chain`)**:
    - **Role**: Offers a solution to the challenge presented in the plot. The emphasis is on collaboration and positive actions, which encourages both the child and parent to discuss possible outcomes.
    - **Example Resolution**: If the characters face a problem (e.g., a trapped animal), the agent suggests different approaches to solve it, encouraging creativity and teamwork.

4. **Moral Lesson (`moral_chain`)**:
    - **Role**: Ends the story with a moral, reinforcing values like kindness or perseverance. This part often includes questions that parents can use to discuss the story’s moral with their children.

```python
from langchain.chains import SequentialChain

# Define a sequential chain to link all agents and create the complete story
overall_chain = SequentialChain(
    chains=[setting_chain, plot_chain, resolution_chain, moral_chain],
    input_variables=["context", "story_prompt", "character_details"],
    output_variables=["setting", "plot", "resolution", "moral"]
)
```

- **Sequential Interaction**: The **SequentialChain** links all the agents. Each agent receives input and contributes to the next, ensuring smooth and coherent storytelling.

### 3. **Chainlit Integration for User Interaction**

**Chainlit** is used to manage the interaction between the user and the storytelling agents. It allows for step-by-step communication, guiding users through context creation, character introduction, and eventually story generation.

- **Chat Flow**:
  1. **Chat Start**: When a user starts a conversation, Chainlit initializes the storytelling chain and prompts the user for a story context.
  2. **Context and Character Details**:
     - The user first provides a context/theme.
     - The bot retrieves relevant information using **Chroma** and asks for character details.
  3. **Story Generation**:
     - With both the context and character details, the bot runs the **overall chain**, generating the full story in distinct parts: setting, plot, resolution, and moral.
  4. **Interactive Output**: The story is then presented to the user in Bulgarian, ensuring it is engaging and culturally appropriate.

```python
import chainlit as cl

@cl.on_chat_start
async def start():
    await cl.Message(content="Здравейте! Моля, въведете тема или контекст за вашата приказка.").send()
    chain = overall_chain  # Initialize the chain for story creation
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    user_input = message.content

    if not cl.user_session.get("context"):
        context = retrieve_context(user_input)
        cl.user_session.set("context", context)
        await cl.Message(content="Контекстът е подготвен. Моля, въведете информация за героите.").send()
    else:
        character_details = user_input
        result = await chain.acall({
            "context": cl.user_session.get("context"),
            "story_prompt": "Създайте приказка",
            "character_details": character_details
        })

        # Combine the generated segments into a full story
        story = "\n\n".join([result["setting"], result["plot"], result["resolution"], result["moral"]])
        await cl.Message(content=story).send()
```

- **Session Management**: **Chainlit's session** capabilities ensure that the context and character details are stored during the interaction, providing continuity.

### Summary

1. **Chroma Vectorstore**:
   - Retrieves relevant content for the story based on user input, ensuring that each story is unique and contextually rich.

2. **Storytelling Agents**:
   - **Setting, Plot, Resolution, Moral**: Each agent is guided by specialized prompts to handle distinct parts of the story. The agents are interconnected using a sequential chain, ensuring the story flows seamlessly.

3. **Chainlit Integration**:
   - Manages user interaction, prompting for story context and character details. It uses these inputs to run the storytelling agents sequentially, creating a personalized and interactive story.

### Interactive and Unique Experience

- **Unique Story Creation**: Each story is uniquely generated using user inputs and context from the Chroma vectorstore, ensuring a personalized experience every time.
- **Step-by-Step Interaction**: The bot involves the user in decision-making, encouraging participation by developing a challenge and resolution based on their choices.
- **Moral and Educational Value**: The final moral of the story helps parents discuss important values like creativity, kindness, and teamwork with their children.

This combination of **retrieving personalized context**, **engaging agents** to develop story segments, and **interactive user sessions** makes the Bulgarian Fairy Tale Bot a powerful and educational tool for parents and children.
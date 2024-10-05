# Installation and Running Instructions

## Overview

This guide provides step-by-step instructions on how to set up and run the `fairy_tale_generator_agents_BG.py` script, which creates Bulgarian fairy tales based on user input. The script leverages Chainlit, Langchain, and various other libraries to process and generate content interactively. It uses environment variables and ChromaDB to handle vector embeddings.

### Prerequisites

1. **Python Version**: Ensure that you have Python 3.9 or later installed.
2. **Conda Environment**: It's recommended to use a virtual environment (e.g., conda) to manage dependencies for this script.  env.yml is provided in reposserot root.
3. **Git**: Ensure you have Git installed for version control.

### Step-by-Step Setup

#### Step 1: Clone the Repository

First, clone the repository from GitHub where your script is located:

```sh
git clone https://github.com/vjfrosty/Bulgarian-AI-Folktales.git
```

#### Step 2: Create a Virtual Environment

Navigate to the project directory and create a conda environment:

```sh
cd Bulgarian-AI-Folktales
conda env create -f env.yml -n Embed
```

Activate the environment:

```sh
conda activate Embed
```


```

#### Step 4: Set Up Environment Variables

The script requires an API key for OpenAI. You need to store this API key in a configuration file or set it as an environment variable. Here, we store it in a JSON file named `config.json`:

1. **Create `config.json` in the root directory**:

   ```json
   {
       "OPENAI_API_KEY": "your_openai_api_key_here"
   }
   ```

2. **Load the Environment Variables**:

   Ensure that you have a `.env` file if you want to use other environment variables.

#### Step 5: Prepare the Chroma Directory

Unzip Basic_chroma.zip to ../Base_Embeded, Make sure that the directory `../Base_Embeded/Basic_chroma` exists, or modify the `CHROMA_PATH` in the script to the correct location where your ChromaDB data will be stored. 

#### Step 6: Run the Script with Chainlit

After setting up the environment, run the script using Chainlit to start the fairy tale generator bot:

1. **Navigate to the Project Directory**:

   ```sh
   cd c:\GitHub\Bulgarian-AI-Folktales\app
   ```

2. **Run Chainlit**:

   Run the Chainlit app by executing the command:

   ```sh
   chainlit run --port 9999 fairy_tale_generator_agents_BG.py -w
   ```

   - `--port 9999`: Specifies the port to run Chainlit on. You can change this as needed.
   - `-w`: Enables auto-reload, which allows you to make changes to your script without stopping the server.

### Usage Instructions

Once the Chainlit server is up and running:

1. **Open Your Browser**: Go to `http://localhost:9999` to interact with the fairy tale generator.
2. **Start the Interaction**: You will be greeted with a welcome message prompting you to provide a context or theme for your story.
3. **Provide Input**: Enter the context and main characters to start the story generation process. The chatbot will ask additional questions to create a comprehensive narrative, including setting, plot development, conflict resolution, and moral lessons.
4. **Story Creation**: The bot will generate each segment of the story based on the user-provided information, ensuring an engaging narrative suitable for children.

### Notes

1. **ChromaDB Path**: The `CHROMA_PATH` is set to `"../Base_Embeded/Basic_chroma"`. Ensure this directory is accessible and writable.
2. **API Key Security**: Make sure that your API key is kept private and not shared publicly.
3. **Story Output Language**: The prompts are set to generate text in **Bulgarian**, which is important for localization purposes.

### Common Issues and Troubleshooting

- **Missing Dependencies**: Ensure all Python packages are installed in the active environment.
- **API Key Issues**: Verify that your OpenAI API key is correctly loaded from `config.json` and set as an environment variable.
- **Port Issues**: If port 9999 is already in use, try using a different port by changing the `--port` option.

### Further Customization

- **Prompt Templates**: Modify the `PromptTemplate` definitions to adjust the story creation process and tailor the narrative style.
- **Embedding Models**: You can experiment with different embedding models or change the parameters like `temperature` for more creative responses.
- **Language**: The current prompts are specifically tailored for **Bulgarian**. Modify the prompts if you wish to generate content in other languages.

### Example

Run the following to start the chatbot:

```sh
conda activate Embed
cd c:\GitHub\Bulgarian-AI-Folktales\app
chainlit run --port 9999 fairy_tale_generator_agents_BG.py -w
```

Then open `http://localhost:9999` in your browser to interact with the bot and start creating a Bulgarian fairy tale.

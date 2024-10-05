Here's a markdown description of the `embed_documents` function and its related components:

```markdown
# Document Embedding with Contextual Enhancement

This system embeds documents with contextual information and stores them in a Chroma vector database. It handles both small (60-1000 tokens) and large (1000+ tokens) documents differently.

## Key Components

- `TEST_CHROMA_PATH`: Path to store the Chroma database
- `TPM_LIMIT`: Token-per-minute limit (200,000)
- `PAUSE_TIME`: Pause duration when approaching TPM limit (80 seconds)
- `OpenAIEmbeddings`: Used for creating embeddings (model: "text-embedding-3-small")
- `ChatOpenAI`: Language model for generating context (model: "gpt-4o-mini")

## Main Function: `embed_documents`

```python
def embed_documents(documents: List[Tuple[Document, int]], llm: ChatOpenAI, embeddings, chroma_path: str) -> None
```

### Parameters:
- `documents`: List of (Document, token_count) tuples
- `llm`: ChatOpenAI instance for generating context
- `embeddings`: Embedding model
- `chroma_path`: Path to store the Chroma database

### Functionality:

1. **TPM Limit Handling**: Pauses processing when approaching the token-per-minute limit.

2. **Document Size Handling**:
   - Small documents (60-1000 tokens): Processed as a single unit.
   - Large documents (1000+ tokens): Split into chunks of ~1000 tokens with 200 token overlap.

3. **Contextual Embedding**:
   - Small documents: Uses `get_contextual_embedding()` with `prompt_type="enhancing"`.
   - Large documents: 
     - Applies `get_contextual_embedding()` with `prompt_type="contextual"` to each chunk.
     - Generates document-level enhancement using `prompt_type="enhancing"`.

4. **Content Combination**:
   - Small documents: Combines original content with enhancing information.
   - Large documents: Combines chunk content with chunk-specific contextual info and document-level enhancing info.

5. **Chroma DB Storage**: Creates and persists the Chroma database with the processed documents.

## Helper Function: `get_contextual_embedding`

```python
def get_contextual_embedding(doc: Document, llm: ChatOpenAI, prompt_type: str, full_text: str = None) -> str
```

Generates contextual or enhancing embeddings based on the `prompt_type`:
- "contextual": Provides context for a chunk within the full document.
- "enhancing": Generates a concise summary of the story, including synopsis, characters, setting, and moral.

## Usage

```python
small_docs = binned_documents['60-1000'][:1]
large_docs = [binned_documents['1000-inf'][-1]]
test_docs = small_docs + large_docs

embed_documents(test_docs, llm, embeddings, TEST_CHROMA_PATH)
```

This system enhances document embeddings with contextual information, improving search retrieval capabilities for both small and large documents.
```

This markdown description provides a comprehensive overview of the `embed_documents` function, its components, and how it processes different types of documents. It can serve as documentation for users who want to understand and use this embedding system.

### Books parser

The `split_file_to_json` script reads a given text file, processes its content, and splits it into chunks, which are then saved as a JSON file. The chunks are categorized based on the content structure and formatted with specific tags such as `"author"` or `"story"`. This script is useful for converting books or texts with multiple stories into structured JSON format for further processing or analysis.
This function reads an input text file, processes the content to remove tags and clean text, and writes the output to a JSON file with a specific structure.

### Parameters

- `input_file` (str): Path to the input text file.
- `fixed_text` (str): The title of the book. This is used as the `"book"` field in the output JSON.
- `author` (str): The name of the author. It is cleaned to remove non-printable characters and extra spaces, then used as metadata in the JSON.
- `output_folder` (str): Path to the directory where the output JSON file will be saved. If it does not exist, it will be created.

### Description

1. **Create Output Directory**: Ensures the provided output directory exists, creating it if necessary.
2. **Set Output File Path**: Derives the output JSON filename from the input text file name.
3. **Clean Author Field**: Uses regular expressions to remove non-printable characters, leaving only alphabetic characters, numbers, spaces, and punctuation.
4. **Read Input File**: Reads the entire content of the input file line by line.
5. **Remove Tags**: Removes all tags of the format `{tag}` from each line using a regular expression.
6. **Build JSON Structure**:
    - The output JSON contains the book title, author, and a list of stories.
    - Stories are derived from splitting the content by every second newline. Each story has a title and content field, formatted based on its position (e.g., "author" for the first story, "story" for subsequent ones).
7. **Write Output File**: Serializes the processed data into a JSON file, formatted for readability.
8. **Print Output Location**: Provides feedback on where the JSON file has been saved.

### JSON Structure

The JSON output has the following structure:

```json
{
    "book": "<fixed_text>",
    "author": "<author>",
    "stories": [
        {
            "author": "<story_title>",
            "content": "<story_content>"
        },
        {
            "story": "<story_title>",
            "content": "<story_content>"
        },
        {
            "id": "<story_title>",
            "sources": "<story_content>"
        }
    ]
}
```
- The `"book"` and `"author"` fields contain the metadata for the input text.
- `"stories"` is a list of dictionaries, where each dictionary represents a chunk of text.
- The tags used for the `"stories"` depend on the position of the story within the text (e.g., `"author"`, `"story"`, `"id"`, and `"sources"`).

## Output

- **JSON File**: The output JSON file will be saved in the specified output folder with the same name as the input text file but with a `.json` extension.
- **Print Statements**: Each processed file will result in a message that indicates the location of the saved JSON file.

## Example Usage

To run the script for a collection of books, modify the list of dictionaries (`data`) with details for each text file, then call the `split_file_to_json` function in a loop:

```python
# Example usage with a loop
data = [
    {"input_file": "Velichko_Vylchev_-_Hityr_Petyr_-_1484-b.txt", "book_name": "Хитър Петър - НАРОДНИТЕ АНЕКДОТИ ЗА ХИТЪР ПЕТЪР", "author": "Величко Вълчев"},
    {"input_file": "Sava_Popov_-_Hityr_Petyr_-_1503-b.txt", "book_name": "Хитър Петър", "author": "Сава Попов"},
    # Additional entries here...
]

for entry in data:
    directory = "books"
    input_file = os.path.join(directory, entry["input_file"])
    book_name = entry["book_name"]
    author = entry["author"]
    output_folder = 'output_json_files'
    
    print(f"Processing file: {input_file}, Book: {book_name}, Author: {author}")
    split_file_to_json(input_file, book_name, author, output_folder)
```

## Notes

1. **File Encoding**: The input file should be encoded in UTF-8 to ensure proper handling of non-ASCII characters.
2. **Output Structure**: Each chunk in the text file is split based on every second newline and treated as a distinct story.
3. **Cleaning the Text**: All non-printable characters are removed, and text is cleaned to ensure consistency across output files.

## Error Handling

- The script assumes the provided input file exists and is accessible. If the file path is incorrect or the file is missing, the script will throw a `FileNotFoundError`.
- If the output folder cannot be created due to permission issues, an `OSError` will be thrown.


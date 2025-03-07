# Autonomous Agentic PDF Q/A Tool

**A Python-based LLM Agent for Question Answering on PDF Documents**

This project implements a Local Large Language Model (LLM) agent capable of answering questions based on the content of a provided PDF document. It leverages various LLM providers (like Gemini and OpenAI) and embedding techniques to understand and respond to user queries.

## Features

*   **PDF Document Processing:** Loads and chunks text content from PDF files.
*   **Multiple LLM Support:** Integrates with both Google's Gemini and OpenAI models.
*   **Flexible Embedding Options:** Supports Sentence Transformers and OpenAI embedding models.
*   **Vector Store for Retrieval:** Utilizes FAISS for efficient similarity search of document chunks.
*   **Reranking for Relevance:** Employs a CrossEncoder model to refine the selection of relevant context.
*   **Command-Line Interface (CLI):** Provides a user-friendly CLI for executing the application with different configurations.
*   **JSON Output:** Presents the question-answering results in a structured JSON format.

## File Structure

```
llm-agent-pdf-reader/
├── .gitignore
├── README.md
├── run_agent.py        # CLI script with configurable LLM and embedding
├── requirements.txt    # Project dependencies
└── src/
    ├── __init__.py
    ├── config.py       # Configuration settings
    ├── document_processing/
    │   ├── __init__.py
    │   └── pdf_processor.py  # Handles PDF loading and chunking
    ├── embedding/
    │   ├── __init__.py
    │   └── embedding_models.py # Implements embedding models
    ├── llm/
    │   ├── __init__.py
    │   └── llm_agents.py     # Interfaces with LLM providers
    ├── question_answering/
    │   ├── __init__.py
    │   └── qa_agent.py       # Orchestrates question answering
    ├── retrieval/
    │   ├── __init__.py
    │   └── vector_store.py   # Implements the vector store
    └── utils/
        ├── __init__.py
        └── logger.py         # Sets up logging
```

## Installation

1. **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd llm-agent-pdf-reader
    ```

2. **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1. **Environment Variables:** The application relies on environment variables for API keys and other settings. You can create a `.env` file in the project's root directory and populate it with the necessary values. Example `.env` file:

    ```env
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_MODEL_NAME="text-embedding-3-large"
    LLM_MODEL_NAME="gemini-1.5-flash"
    OPENAI_LLM_MODEL_NAME="gpt-4o-mini"
    RERANKER_MODEL_NAME="cross-encoder/ms-marco-MiniLM-L-12-v2"
    PDF_PATH="documents/handbook.pdf"
    ```

    **Note:**
    *   Obtain your API keys from Google AI Studio for Gemini and OpenAI Platform for OpenAI.
    *   You can customize the model names and default PDF path in the `.env` file.

## Usage

The primary way to run the application is through the `run_agent.py` script, which provides a flexible command-line interface.

```bash
python run_agent.py --help
```

This will display the available command-line options:

```
usage: run_agent.py [-h] [--pdf_path PDF_PATH]
                    [--llm_model {gemini,openai}]
                    [--embedding_model {sentence-transformers,openai}]
                    [--questions QUESTIONS] [--questions_file QUESTIONS_FILE]

LLM Agent PDF Reader CLI

options:
  -h, --help            show this help message and exit
  --pdf_path PDF_PATH   Path to the PDF file.
  --llm_model {gemini,openai}
                        Choose the LLM model (gemini or openai).
  --embedding_model {sentence-transformers,openai}
                        Choose the embedding model (sentence-transformers or
                        openai).
  --questions QUESTIONS
                        Semicolon-separated string of questions.
  --questions_file QUESTIONS_FILE
                        Path to a file containing questions (one question per
                        line).
```

**Examples:**

*   **Run with default settings (uses Gemini and default questions):**
    ```bash
    python run_agent.py
    ```

*   **Specify a PDF file:**
    ```bash
    python run_agent.py --pdf_path path/to/your/document.pdf
    ```

*   **Use OpenAI as the LLM:**
    ```bash
    python run_agent.py --llm_model openai
    ```

*   **Use OpenAI for embeddings:**
    ```bash
    python run_agent.py --embedding_model openai
    ```

*   **Provide questions directly:**
    ```bash
    python run_agent.py --questions "What are the key benefits?, Explain the company culture."
    ```

*   **Read questions from a file:**
    ```bash
    python run_agent.py --questions_file my_questions.txt
    ```

*   **Combine options:**
    ```bash
    python run_agent.py --pdf_path another_document.pdf --llm_model openai --embedding_model openai --questions "What is the remote work policy?"
    ```

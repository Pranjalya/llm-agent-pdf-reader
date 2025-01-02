import argparse
import json
import os
from src.config import Settings
from src.document_processing.pdf_processor import PDFProcessor
from src.embedding.embedding_models import SentenceTransformersEmbedding, OpenAIEmbedding
from src.llm.llm_agents import GeminiLLMAgent, OpenAILLMAgent
from src.retrieval.vector_store import FAISSVectorStore
from src.question_answering.qa_agent import QuestionAnswerer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
settings = Settings()

def load_questions(questions_arg, questions_file_arg):
    """Loads questions from either a string or a file."""
    if questions_arg:
        questions = [q.strip() for q in questions_arg.split(';')]
        return [q for q in questions if len(q)]
    elif questions_file_arg:
        try:
            with open(questions_file_arg, 'r') as f:
                questions_input = f.read()
                questions = [q.strip() for q in questions_input.splitlines()]
                return [q for q in questions if len(q)]
        except FileNotFoundError:
            logger.error(f"Questions file not found: {questions_file_arg}")
            return None
    else:
        default_questions_input = """
        What is the name of the company?
        Who is the CEO of the company?
        What is their vacation policy?
        What is the termination policy?
        What is the meaning of Zania and why is it named so?
        """
        questions = [q.strip() for q in default_questions_input.split('\n')]
        return [q for q in questions if len(q)]

def main():
    parser = argparse.ArgumentParser(description="LLM Agent PDF Reader CLI")
    parser.add_argument("--pdf_path", type=str, default=settings.PDF_PATH, help="Path to the PDF file.")
    parser.add_argument(
        "--llm_model",
        type=str,
        choices=['gemini', 'openai'],
        default='gemini',
        help="Choose the LLM model (gemini or openai).",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        choices=['sentence-transformers', 'openai'],
        default='sentence-transformers',
        help="Choose the embedding model (sentence-transformers or openai).",
    )
    parser.add_argument(
        "--questions",
        type=str,
        help="Semicolon-separated string of questions.",
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        help="Path to a file containing questions (one question per line).",
    )

    args = parser.parse_args()

    pdf_file = args.pdf_path
    llm_model_name = args.llm_model
    embedding_model_name = args.embedding_model
    questions = load_questions(args.questions, args.questions_file)

    if questions is None:
        return

    pdf_processor = PDFProcessor(pdf_file)
    try:
        chunks = pdf_processor.load_and_chunk()
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_file}")
        return
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return

    if not chunks:
        logger.warning("No text chunks were extracted from the PDF.")
        return

    # Initialize Embedding Model
    embedding_model = None
    if embedding_model_name == 'sentence-transformers':
        embedding_model = SentenceTransformersEmbedding(settings.EMBEDDING_MODEL_NAME)
    elif embedding_model_name == 'openai':
        if not settings.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY environment variable not set.")
            return
        embedding_model = OpenAIEmbedding(settings.OPENAI_API_KEY, settings.OPENAI_EMBEDDING_MODEL_NAME)
    else:
        logger.error(f"Invalid embedding model: {embedding_model_name}")
        return

    embeddings = embedding_model.encode(chunks)

    # Initialize Vector Store
    vector_store = FAISSVectorStore(embeddings.shape[1])
    vector_store.add_embeddings(chunks, embeddings)

    # Initialize LLM Agent
    llm_agent = None
    if llm_model_name == 'gemini':
        if not settings.GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY environment variable not set.")
            return
        llm_agent = GeminiLLMAgent(api_key=settings.GEMINI_API_KEY)
    elif llm_model_name == 'openai':
        if not settings.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY environment variable not set.")
            return
        llm_agent = OpenAILLMAgent(api_key=settings.OPENAI_API_KEY, model_name=settings.OPENAI_LLM_MODEL_NAME)
    else:
        logger.error(f"Invalid LLM model: {llm_model_name}")
        return

    qa_agent = QuestionAnswerer(embedding_model, llm_agent, vector_store, pdf_processor)
    results = qa_agent.answer_questions(questions)

    # Output as JSON
    print("\nJSON Output:")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
import os
import json
from src.config import settings
from src.document_processing.pdf_processor import PDFProcessor
from src.embedding.embedding_models import SentenceTransformersEmbedding
from src.llm.llm_agents import GeminiLLMAgent
from src.retrieval.faiss_store import FAISSVectorStore
from src.question_answering.qa_agent import QuestionAnswerer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    pdf_file = settings.PDF_PATH
    questions_input = """
    What is the name of the company?
    Who is the CEO of the company?
    What is their vacation policy?
    What is the termination policy?
    What is the meaning of Zania and why is it named so?
    """
    questions = [q.strip() for q in questions_input.split('\n')]
    questions = [q for q in questions if len(q)]

    if not settings.GEMINI_API_KEY:
        logger.error("GOOGLE_API_KEY environment variable not set.")
        return

    pdf_processor = PDFProcessor(pdf_file)
    chunks = pdf_processor.load_and_chunk()

    embedding_model = SentenceTransformersEmbedding(settings.EMBEDDING_MODEL_NAME)
    embeddings = embedding_model.encode(chunks)

    # Initialize Vector Store
    vector_store = FAISSVectorStore(embeddings.shape[1])
    vector_store.add_embeddings(chunks, embeddings)

    llm_agent = GeminiLLMAgent(api_key=settings.GEMINI_API_KEY)

    qa_agent = QuestionAnswerer(embedding_model, llm_agent, vector_store, pdf_processor)
    results = qa_agent.answer_questions(questions)

    for result in results:
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print("-" * 50)

    # Output as JSON
    print("\nJSON Output:")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
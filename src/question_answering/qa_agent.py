import logging
from typing import List, Dict

from sentence_transformers import CrossEncoder

from src.config import settings
from src.embedding.embedding_models import EmbeddingModel
from src.llm.llm_agents import LLMAgentInterface
from src.retrieval.vector_store import VectorStore
from src.document_processing.pdf_processor import PDFProcessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class QuestionAnswerer:
    def __init__(self, embedding_model: EmbeddingModel, llm_agent: LLMAgentInterface, vector_store: VectorStore, pdf_processor: PDFProcessor,
                 reranker_model_name: str = settings.RERANKER_MODEL_NAME, top_n: int = 10, rerank_top_n: int = 5):
        self.embedding_model = embedding_model
        self.llm_agent = llm_agent
        self.vector_store = vector_store
        self.pdf_processor = pdf_processor
        self.reranker = CrossEncoder(reranker_model_name)
        self.top_n = top_n
        self.rerank_top_n = rerank_top_n

    def answer_question(self, question: str) -> Dict:
        query_embedding = self.embedding_model.encode([question])[0]
        relevant_indices_with_scores = self.vector_store.search(query_embedding, top_k=self.top_n)

        if not relevant_indices_with_scores:
            return {"question": question, "answer": "Data Not Available"}

        relevant_chunks = [self.pdf_processor.get_chunk(index) for index, _ in relevant_indices_with_scores]
        relevant_chunk_texts = [self.vector_store.get_text(index) for index, _ in relevant_indices_with_scores]

        # Re-ranking with CrossEncoder
        rerank_pairs = [(question, chunk) for chunk in relevant_chunk_texts]
        rerank_scores = self.reranker.predict(rerank_pairs)

        # Sort chunks based on reranker scores
        ranked_chunks_with_scores = sorted(zip(relevant_chunks, rerank_scores), key=lambda x: x[1], reverse=True)
        top_ranked_chunks = [chunk for chunk, _ in ranked_chunks_with_scores[:self.rerank_top_n]]

        top_rated_chunk_texts = list(map(self.pdf_processor.get_origin_text_from_chunk, top_ranked_chunks))

        context = "\n\n".join(top_rated_chunk_texts)

        prompt = f"""You are an expert assistant designed to answer questions based on the provided document content.

        **Document Content:**
        -------------------------------------------
        {context}
        -------------------------------------------

        **Question:** {question}

        **Instructions:**
        1. Carefully read the document content to find the answer to the question.
        2. If the answer is explicitly stated in the document, provide the exact answer verbatim.
        3. If the answer is not explicitly stated but can be inferred from the document, explain the answer based on the provided content.
        4. If the document does not contain information to answer the question, state: "Data Not Available"
        5. Do not make up information or provide answers that are not supported by the document content.

        **Answer:**
        """
        llm_response = self.llm_agent.query(prompt)
        response = llm_response if llm_response else "Data Not Available"
        return {"question": question, "answer": response}

    def answer_questions(self, questions: List[str]) -> List[Dict]:
        results = []
        for question in questions:
            answer = self.answer_question(question)
            results.append(answer)
        return results
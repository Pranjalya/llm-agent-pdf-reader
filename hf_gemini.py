import json
import logging
import os
import random
import numpy as np
from typing import List, Dict

from dotenv import load_dotenv
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.types.doc.document import TextItem
from sentence_transformers import SentenceTransformer, CrossEncoder

import google.generativeai as genai
import faiss

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class PDFProcessor:
    """Handles PDF document processing."""
    def __init__(self, pdf_path: str, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.pdf_path = pdf_path
        self.embed_model = embed_model
        tokenizer = AutoTokenizer.from_pretrained(embed_model)
        self.chunker = HybridChunker(tokenizer=tokenizer)
        self.text_content = self._extract_text()

    def _extract_text(self) -> str:
        """Extracts text content from the PDF."""
        doc = None
        try:
            conv_res = DocumentConverter().convert("handbook.pdf")
            doc = conv_res.document
            logging.info(f"Successfully extracted text from {self.pdf_path}")
            return doc
        except FileNotFoundError:
            logging.error(f"PDF file not found at: {self.pdf_path}")
            raise
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {e}")
            raise

    def chunk_text(self) -> List[str]:
        """Splits the text into smaller chunks."""
        chunk_iter = self.chunker.chunk(dl_doc=self.text_content)
        chunks = list(chunk_iter)
        return chunks

    def get_origin_text_from_chunk(self, chunk):
        text = " ".join([doc_item.orig for doc_item in chunk.meta.doc_items if type(doc_item) is TextItem])
        if len(text.strip()) == 0:
            text = chunk.text
        return text


class LLMAgent:
    """Handles interactions with the LLM."""
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def query_llm(self, prompt: str) -> str:
        """Queries the LLM with a given prompt."""
        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            logging.info(f"LLM Response: {answer[:100]}...")  # Log a longer snippet
            return answer
        except Exception as e:
            logging.error(f"Error querying LLM: {e}")
            return "Error processing the question."


class QuestionAnswerer:
    def __init__(self, pdf_processor: PDFProcessor, llm_agent: LLMAgent,
                 threshold: float = 0.3, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.pdf_processor = pdf_processor
        self.llm_agent = llm_agent
        embed_model = self.pdf_processor.embed_model
        if embed_model.startswith("sentence-transformers"):
            embed_model = embed_model.replace("sentence-transformers/", "")
        self.embedding_model = SentenceTransformer(embed_model)
        self.reranker = CrossEncoder(reranker_model)
        self.chunks, self.embeddings = self._embed_chunks()
        self.faiss_index = self._build_faiss_index(self.embeddings)
        self.threshold = threshold

    def _embed_chunks(self):
        chunks = self.pdf_processor.chunk_text()
        chunk_texts = [f"Title : {' '.join(chunk.meta.headings)} | {chunk.text}".replace("\n", " ") for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts)
        return chunks, embeddings

    def _build_faiss_index(self, embeddings: np.ndarray):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # Consider IndexIVFFlat for larger datasets
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return index

    def _get_relevant_chunks(self, question: str, top_n: int = 10, rerank_top_n: int = 5):
        question_embedding = self.embedding_model.encode([question])[0]
        faiss.normalize_L2(np.array(question_embedding).reshape(1, -1))
        distances, indices = self.faiss_index.search(np.array(question_embedding).reshape(1, -1), top_n)

        # Fetch initial relevant chunks
        initial_relevant_chunks = [(self.chunks[i], distances[0][idx]) for idx, i in enumerate(indices[0]) if i != -1]
    
        if not initial_relevant_chunks:
            return []

        # Re-ranking with CrossEncoder
        rerank_pairs = [(question, f"Title : {' '.join(chunk.meta.headings)} | {chunk.text}".replace("\n", " ")) for chunk, _ in initial_relevant_chunks]
        rerank_scores = self.reranker.predict(rerank_pairs)

        # Sort chunks based on reranker scores
        ranked_chunks_with_scores = sorted(zip(initial_relevant_chunks, rerank_scores), key=lambda x: x[1], reverse=True)
        top_ranked_chunks = [item[0][0] for item in ranked_chunks_with_scores[:rerank_top_n]]

        return top_ranked_chunks

    def answer_question(self, question: str) -> str:
        relevant_chunks = self._get_relevant_chunks(question)

        if not relevant_chunks:
            return {
                "question": question,
                "answer": "Data Not Available"
            }

        # Fetch the original entire text that chunk is part of
        context = "\n\n".join([self.pdf_processor.get_origin_text_from_chunk(relevant_chunk) for relevant_chunk in relevant_chunks])

        # Improved Prompt
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
        llm_response = self.llm_agent.query_llm(prompt)
        response = llm_response if llm_response else "Data Not Available"
        return {
                "question": question,
                "answer": response
            }

    def answer_questions(self, questions: List[str]) -> List:
        results = []
        for question in questions:
            answer = self.answer_question(question)
            results.append(answer)
        return results


if __name__ == "__main__":
    pdf_file = "handbook.pdf"

    questions_input = f"""
    What is the name of the company?
    Who is the CEO of the company?
    What is their vacation policy?
    What is the termination policy?
    What is the meaning of Zania and why is it named so?
    """
    questions = [q.strip() for q in questions_input.split('\n')]
    questions = [q for q in questions if len(q)]

    google_api_key = os.environ.get("GEMINI_API_KEY")
    if not google_api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")

    pdf_processor = PDFProcessor(pdf_file)
    llm_agent = LLMAgent(api_key=google_api_key)

    qa_agent = QuestionAnswerer(pdf_processor, llm_agent)
    results = qa_agent.answer_questions(questions)

    for result in results:
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print("-" * 50)
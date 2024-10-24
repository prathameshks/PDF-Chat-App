# app.py
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import fitz
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import time
import os
import gc
from typing import List, Dict, Tuple
import tempfile
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

class ModelManager:
    def __init__(self):
        login(os.getenv('HF_TOKEN'))
        
    @st.cache_resource
    def load_embedding_model():
        """Cache the embedding model in memory"""
        return SentenceTransformer('all-mpnet-base-v2')
    
    @st.cache_resource
    def load_llm_model():
        """Cache the LLM model and tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        return tokenizer, model

class PDFProcessor:
    def __init__(self):
        self.chunk_size = 10
        self.overlap_sentences = 2
        self.min_chunk_size = 100
        
    @st.cache_data
    def extract_text_from_pdf(_self, file_content: bytes) -> List[Dict]:
        """Cache PDF text extraction results"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            doc = fitz.open(tmp_file_path)
            page_list = []
            
            # Create progress bar
            progress_bar = st.progress(0)
            total_pages = len(doc)
            
            for page_no, page in enumerate(doc):
                text = page.get_text().replace('\n', '').strip()
                page_list.append({
                    'page_no': page_no,
                    'text': text,
                    'char_count': len(text),
                    'token_count': len(text)/4
                })
                # Update progress
                progress_bar.progress((page_no + 1) / total_pages)
            
            return page_list
        finally:
            os.unlink(tmp_file_path)
    
    @st.cache_data
    def make_chunks(_self, page_list: List[Dict]) -> List[Dict]:
        """Cache chunking results"""
        nlp = spacy.blank('en')
        nlp.add_pipe('sentencizer')
        
        chunk_list = []
        total_pages = len(page_list)
        progress_bar = st.progress(0)
        
        for idx, page in enumerate(page_list):
            # Process sentences
            doc = nlp(page['text'])
            sentences = [sent.text for sent in doc.sents]
            
            # Create chunks
            for i in range(0, len(sentences), _self.chunk_size - _self.overlap_sentences):
                chunk = sentences[i:i + _self.chunk_size]
                chunk_text = ' '.join(chunk)
                
                if len(chunk_text) >= _self.min_chunk_size:
                    chunk_list.append({
                        'page_no': page['page_no'],
                        'text': chunk_text,
                        'sentence_count': len(chunk),
                        'char_count': len(chunk_text),
                        'token_count': len(chunk_text)/4
                    })
            
            progress_bar.progress((idx + 1) / total_pages)
        
        return chunk_list

class EmbeddingManager:
    def __init__(self, model):
        self.model = model
        self.batch_size = 32
        
    @st.cache_data
    def create_embeddings(_self, chunk_list: List[Dict]) -> Tuple[torch.Tensor, List[Dict]]:
        """Cache embeddings creation results"""
        chunks = [chunk['text'] for chunk in chunk_list]
        total_chunks = len(chunks)
        progress_bar = st.progress(0)
        all_embeddings = []
        
        for i in range(0, total_chunks, _self.batch_size):
            batch = chunks[i:i + _self.batch_size]
            batch_embeddings = _self.model.encode(batch, convert_to_tensor=True)
            all_embeddings.extend(batch_embeddings)
            progress_bar.progress((i + len(batch)) / total_chunks)
        
        embeddings_tensor = torch.stack(all_embeddings)
        return embeddings_tensor, chunk_list
    
    @st.cache_data
    def save_embeddings(_self, chunk_list: List[Dict], file_name: str):
        """Cache embeddings saving results"""
        df = pd.DataFrame(chunk_list)
        df.to_csv(file_name, index=False)
        return file_name

class QASystem:
    def __init__(self, embedding_model, llm_model, tokenizer):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        
    def get_relevant_context(self, query: str, embeddings: torch.Tensor, chunk_list: List[Dict], k: int = 5):
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        dot_scores = torch.matmul(embeddings, query_embedding)
        top_k_indices = torch.topk(dot_scores, k=k).indices
        
        context_items = []
        for idx in top_k_indices:
            context_items.append(chunk_list[idx])
        return context_items
    
    def generate_answer(self, query: str, context_items: List[Dict]) -> str:
        context = "\n".join([item['text'] for item in context_items])
        prompt = f"""Based on the following context, answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question:
{query}

Answer:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()

def main():
    st.set_page_config(page_title="PDF Question Answering System", layout="wide")
    
    # Initialize session state
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'chunk_list' not in st.session_state:
        st.session_state.chunk_list = None
    
    # Load models
    with st.spinner("Loading models..."):
        embedding_model = ModelManager.load_embedding_model()
        tokenizer, llm_model = ModelManager.load_llm_model()
        qa_system = QASystem(embedding_model, llm_model, tokenizer)
    
    st.title("PDF Question Answering System")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            # Process PDF
            processor = PDFProcessor()
            page_list = processor.extract_text_from_pdf(uploaded_file.read())
            chunk_list = processor.make_chunks(page_list)
            
            # Create embeddings
            embedding_manager = EmbeddingManager(embedding_model)
            embeddings, chunk_list = embedding_manager.create_embeddings(chunk_list)
            
            # Store in session state
            st.session_state.embeddings = embeddings
            st.session_state.chunk_list = chunk_list
            
            # Save embeddings option
            if st.button("Save Embeddings"):
                file_name = f"embeddings_{int(time.time())}.csv"
                saved_file = embedding_manager.save_embeddings(chunk_list, file_name)
                st.download_button(
                    "Download Embeddings",
                    saved_file,
                    file_name,
                    "text/csv"
                )
    
    # Q&A Interface
    if st.session_state.embeddings is not None:
        query = st.text_input("Ask a question about your document:")
        if query:
            with st.spinner("Generating answer..."):
                context_items = qa_system.get_relevant_context(
                    query,
                    st.session_state.embeddings,
                    st.session_state.chunk_list
                )
                answer = qa_system.generate_answer(query, context_items)
                st.markdown("### Answer:")
                st.write(answer)
                
                # Show relevant context
                if st.checkbox("Show relevant context"):
                    st.markdown("### Relevant Context:")
                    for item in context_items:
                        st.write(f"Page {item['page_no'] + 1}:", item['text'])

if __name__ == "__main__":
    main()
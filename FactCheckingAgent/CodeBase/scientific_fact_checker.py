
#!/usr/bin/env python3
"""
Scientific Fact Checking System
A comprehensive tool for extracting, validating, and fact-checking scientific statements
using Google Gemini AI and Tavily search integration.

Author: Scientific Fact Checking Team
Version: 1.0.0
"""

import os
import json
import re
import time
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import google.generativeai as genai
import pandas as pd
import numpy as np
from tavily import TavilyClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fact_checker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the fact checking system"""
    google_api_key: str
    tavily_api_key: str
    search_domains: List[str]
    batch_size: int
    chunk_size: int
    overlap: int = 50

class SimpleRAGSystem:
    """Simple RAG system using TF-IDF for document retrieval (now metadata-aware)."""

    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[str] = []
        self.chunk_meta: List[Dict[str, Any]] = []  # NEW: keep per-chunk metadata
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.chunk_vectors = None
        self.is_fitted = False

    # -------- existing word-based chunker kept for backward-compat --------
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks (word-based, backward-compatible)."""
        chunks = []
        words = text.split()
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())
        return chunks
    # ----------------------------------------------------------------------

    # NEW: paragraph-aware chunker that targets ~512 'tokens' (approximated by words)
    def _paragraph_chunks(self, text: str, target_words: int = 512, min_words: int = 120) -> List[str]:
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        chunks = []
        buf_words: List[str] = []
        for p in paragraphs:
            pw = p.split()
            # if a single paragraph is large, flush buffer then split paragraph itself
            if len(pw) >= target_words * 1.2:
                if buf_words:
                    buf = ' '.join(buf_words).strip()
                    if len(buf.split()) >= min_words:
                        chunks.append(buf)
                    buf_words = []
                # split this paragraph into ~target_words segments
                for i in range(0, len(pw), target_words):
                    seg = pw[i:i + target_words]
                    if len(seg) >= min_words:
                        chunks.append(' '.join(seg).strip())
                continue

            # accumulate paragraphs
            if len(buf_words) + len(pw) <= target_words:
                buf_words.extend(pw)
            else:
                buf = ' '.join(buf_words).strip()
                if len(buf.split()) >= min_words:
                    chunks.append(buf)
                buf_words = pw[:]  # start new buffer

        if buf_words:
            buf = ' '.join(buf_words).strip()
            if len(buf.split()) >= min_words:
                chunks.append(buf)
        return chunks

    def build_index(self, knowledge_base: str):
        """
        Build the RAG index from a single KB string (legacy behavior, no metadata).
        Kept to avoid breaking existing callers; prefer build_index_from_blocks().
        """
        logger.info("Building RAG index (legacy KB string)...")
        self.chunks = self.chunk_text(knowledge_base)
        self.chunk_meta = [{"source_title": None, "source_url": None} for _ in self.chunks]
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
        self.is_fitted = True
        logger.info("RAG index built successfully (legacy).")

    # NEW: preferred builder with metadata
    def build_index_from_blocks(self, blocks: List[Dict[str, str]]):
        """
        Build index from structured blocks:
        each block = {'text': str, 'source_title': str, 'source_url': str}
        """
        logger.info("Building RAG index from structured blocks (paragraph-aware)...")
        all_chunks, all_meta = [], []
        for b in blocks:
            text = (b.get("text") or "").strip()
            if not text:
                continue
            # paragraph-aware chunking
            p_chunks = self._paragraph_chunks(text, target_words=max(self.chunk_size, 512))
            for ch in p_chunks:
                all_chunks.append(ch)
                all_meta.append({
                    "source_title": b.get("source_title"),
                    "source_url": b.get("source_url"),
                })

        if not all_chunks:
            logger.warning("No chunks produced from blocks; RAG index will be empty.")
            self.chunks, self.chunk_meta = [], []
            self.chunk_vectors, self.is_fitted = None, False
            return

        self.chunks = all_chunks
        self.chunk_meta = all_meta
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
        self.is_fitted = True
        logger.info(f"RAG index built with {len(self.chunks)} chunks (with metadata).")

    def retrieve_relevant_chunks(self, query: str, top_k=3) -> List[str]:
        """Backward-compatible: return only chunk texts (no metadata)."""
        if not self.is_fitted:
            return []
        query_vector = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vector, self.chunk_vectors).flatten()
        top_idx = np.argsort(sims)[-top_k:][::-1]
        return [self.chunks[i] for i in top_idx if sims[i] > 0.1]

    # NEW: return chunks WITH metadata for citation-friendly prompts
    def retrieve_relevant_chunks_with_meta(self, query: str, top_k=3) -> List[Dict[str, Any]]:
        if not self.is_fitted:
            return []
        query_vector = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vector, self.chunk_vectors).flatten()
        top_idx = np.argsort(sims)[-top_k:][::-1]
        out = []
        for i in top_idx:
            if sims[i] > 0.1:
                out.append({
                    "text": self.chunks[i],
                    "source_title": self.chunk_meta[i].get("source_title"),
                    "source_url": self.chunk_meta[i].get("source_url"),
                    "similarity": float(sims[i]),
                })
        return out

    def save_index(self, filepath='rag_index.pkl'):
        """Save the RAG index + metadata to disk"""
        index_data = {
            'chunks': self.chunks,
            'chunk_meta': self.chunk_meta,  # NEW
            'vectorizer': self.vectorizer,
            'chunk_vectors': self.chunk_vectors,
            'is_fitted': self.is_fitted,
            'chunk_size': self.chunk_size,
            'overlap': self.overlap
        }
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        logger.info(f"RAG index saved to {filepath}")

    def load_index(self, filepath='rag_index.pkl'):
        """Load the RAG index from disk"""
        try:
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
            self.chunks = index_data.get('chunks', [])
            self.chunk_meta = index_data.get('chunk_meta', [{"source_title": None, "source_url": None} for _ in self.chunks])
            self.vectorizer = index_data['vectorizer']
            self.chunk_vectors = index_data['chunk_vectors']
            self.is_fitted = index_data['is_fitted']
            self.chunk_size = index_data.get('chunk_size', self.chunk_size)
            self.overlap = index_data.get('overlap', self.overlap)
            logger.info(f"RAG index loaded from {filepath}")
            return True
        except FileNotFoundError:
            logger.info(f"No existing index found at {filepath}")
            return False

class ScientificFactChecker:
    """Main class for scientific fact checking system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.rag_system = SimpleRAGSystem(
            chunk_size=config.chunk_size,
            overlap=config.overlap
        )
        
        # Configure APIs
        genai.configure(api_key=config.google_api_key)
        self.tavily_client = TavilyClient(api_key=config.tavily_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        
    def extract_assertions(self, chapter_path: str) -> List[Dict[str, str]]:
        """Extract testable assertions from a markdown chapter"""
        logger.info(f"Extracting assertions from: {chapter_path}")
        
        try:
            with open(chapter_path, "r", encoding="utf-8") as file:
                chapter_content = file.read()
        except FileNotFoundError:
            logger.error(f"File not found: {chapter_path}")
            return []
        
        # Get chapter name from filename
        chapter_name = Path(chapter_path).stem.replace("_", " ")
        
        # Create extraction prompt
        prompt = self._create_extraction_prompt(chapter_name, chapter_content)
        
        try:
            response = self.gemini_model.generate_content(prompt)
            extracted_data = self._parse_extraction_response(response.text)
            
            # Filter and format testable assertions
            testable_assertions = [
                {
                    "Original Statement": item["original_statement"],
                    "Assertion": item["optimized_assertion"]
                }
                for item in extracted_data if item.get("is_testable", False)
            ]
            
            logger.info(f"Extracted {len(testable_assertions)} testable assertions")
            return testable_assertions
            
        except Exception as e:
            logger.error(f"Error extracting assertions: {e}")
            return []
    
    def _create_extraction_prompt(self, chapter_name: str, content: str) -> str:
        """Create prompt for assertion extraction"""
        return f"""
        # SCIENTIFIC ASSERTION EXTRACTION PROMPT

        ## YOUR ROLE
        You are a SCIENTIFIC FACT-CHECKER tasked with extracting ALL verifiable assertions from scientific textbook content for peer-reviewed validation.

        ## PRIMARY OBJECTIVE
        EXTRACT MAXIMUM NUMBER of scientifically testable assertions from the provided chapter content. Your goal is COMPREHENSIVE COVERAGE - every sentence must be analyzed and NO SENTENCE should be overlooked.

        ## CRITICAL REQUIREMENTS
        - **MANDATORY**: Process EVERY SINGLE SENTENCE in the text
        - **MAXIMIZE EXTRACTION**: Identify as many testable assertions as possible
        - **ZERO OMISSIONS**: Absolutely no sentence should be left unanalyzed

        ## TESTABLE ASSERTION DEFINITION
        A testable assertion is ANY statement that can be verified against peer-reviewed scientific sources:
        - Specific factual claims about properties, mechanisms, or relationships
        - Quantitative data or measurements
        - Cause-and-effect relationships
        - Scientific processes or phenomena descriptions
        - Research findings or experimental results

        **EXCLUDE ONLY**: Pure definitions, introductory phrases, or entirely subjective opinions

        ## ASSERTION OPTIMIZATION FOR SEARCH
        For each testable statement, create an OPTIMIZED ASSERTION that:
        - Contains the key scientific concepts and terminology
        - Uses specific, searchable scientific terms
        - Focuses on the most verifiable claim within the statement
        - Will yield the most relevant peer-reviewed documents when searched
        - Is concise but comprehensive enough for accurate fact-checking

        ## EXTRACTION STEPS
        1. **READ**: Process the text sentence by sentence sequentially
        2. **ANALYZE**: Determine if each sentence contains verifiable scientific content
        3. **EXTRACT**: Pull out the complete sentence containing the assertion
        4. **OPTIMIZE**: Create a search-optimized assertion that will find the most relevant documents

        ## OUTPUT FORMAT
        Return your results in the following JSON format:
        ```json
        [
        {{
            "sentence_number": 1,
            "original_statement": "EXACT sentence from text",
            "is_testable": true,
            "optimized_assertion": "Search-optimized assertion for finding relevant documents"
        }},
        {{
            "sentence_number": 2,
            "original_statement": "EXACT sentence from text",
            "is_testable": false,
            "optimized_assertion": "Search-optimized assertion for finding relevant documents"
        }}
        ]
        ```
        
        **CRITICAL JSON REQUIREMENTS:**
        - Use ONLY double quotes (") for strings, NEVER single quotes
        - Use lowercase true/false for boolean values
        - Ensure every object is separated by commas
        - No trailing commas after the last object
        - All property names must be in double quotes
        - Escape any quotes within text content with backslash (\")
        - Ensure the JSON is valid and parseable

        ## EXECUTION COMMAND
        **Chapter Title**: {chapter_name}

        **Content to Analyze**:
        {content}

        BEGIN COMPREHENSIVE SENTENCE-BY-SENTENCE ANALYSIS NOW.
        """
    
    def _parse_extraction_response(self, response_text: str) -> List[Dict]:
        """Parse the extraction response to extract JSON data"""
        # Remove markdown code formatting if present
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_text = json_match.group(1)
        else:
            # If no code blocks, try to extract what looks like JSON
            json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', response_text)
            if json_match:
                json_text = json_match.group(0)
            else:
                json_text = response_text

        # Clean and fix common JSON issues
        json_text = self._clean_json_text(json_text)
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            logger.error(f"JSON text: {json_text[:500]}...")  # Log first 500 chars for debugging
            
            # Try one more aggressive cleaning approach
            try:
                # Remove all problematic quotes and rebuild
                aggressive_clean = re.sub(r'(?<!\\)"(?![,\]\}:\s])', '\\"', json_text)
                return json.loads(aggressive_clean)
            except json.JSONDecodeError:
                pass
            
            # Try to extract basic information as fallback
            logger.info("Attempting fallback extraction...")
            fallback_data = self._extract_fallback_data(response_text)
            if fallback_data:
                logger.info(f"Fallback extraction successful: {len(fallback_data)} items")
                return fallback_data
            
            return []
    
    def _clean_json_text(self, json_text: str) -> str:
        """Clean and fix common JSON formatting issues"""
        if not json_text:
            return "[]"
        
        # Remove any leading/trailing whitespace
        json_text = json_text.strip()
        
        # Fix common issues with AI-generated JSON
        # 1. Fix missing commas between objects in arrays
        json_text = re.sub(r'}\s*\n\s*{', '},\n{', json_text)
        
        # 2. Fix trailing commas in arrays
        json_text = re.sub(r',\s*]', ']', json_text)
        
        # 3. Fix trailing commas in objects
        json_text = re.sub(r',\s*}', '}', json_text)
        
        # 4. Fix single quotes to double quotes first
        json_text = json_text.replace("'", '"')
        
        # 5. Fix missing quotes around property names (be more careful)
        json_text = re.sub(r'(\s*)([a-zA-Z_]\w*)(\s*):', r'\1"\2"\3:', json_text)
        
        # 6. Simple fix for unescaped quotes - replace problematic quotes in content
        lines = json_text.split('\n')
        fixed_lines = []
        for line in lines:
            if '": "' in line and line.count('"') > 4:  # Line has a string value with potential issues
                parts = line.split('": "', 1)
                if len(parts) == 2:
                    key_part = parts[0] + '": "'
                    value_part = parts[1]
                    if value_part.endswith('",') or value_part.endswith('"'):
                        if value_part.endswith('",'):
                            end_part = '",'
                            value_content = value_part[:-2]
                        else:
                            end_part = '"'
                            value_content = value_part[:-1]
                        value_content = value_content.replace('"', '\\"')
                        line = key_part + value_content + end_part
            fixed_lines.append(line)
        json_text = '\n'.join(fixed_lines)
        
        # 7. Fix boolean values (true/false should be lowercase)
        json_text = re.sub(r'\bTrue\b', 'true', json_text)
        json_text = re.sub(r'\bFalse\b', 'false', json_text)
        
        # 8. Fix null values
        json_text = re.sub(r'\bNone\b', 'null', json_text)
        
        # 9. Remove any comments or extra text
        lines = json_text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('//') and not line.startswith('#'):
                cleaned_lines.append(line)
        
        json_text = '\n'.join(cleaned_lines)
        
        # 10. Ensure the text starts and ends with proper brackets
        if not json_text.startswith('['):
            json_text = '[' + json_text
        if not json_text.endswith(']'):
            json_text = json_text + ']'
        
        return json_text
    
    def _extract_fallback_data(self, response_text: str) -> List[Dict]:
        """Fallback method to extract data when JSON parsing fails"""
        fallback_data = []
        
        # Look for patterns that indicate testable statements
        sentences = re.split(r'[.!?]+', response_text)
        
        sentence_number = 1
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Only consider meaningful sentences
                testable_keywords = [
                    'is', 'are', 'was', 'were', 'has', 'have', 'had',
                    'contains', 'consists', 'includes', 'comprises',
                    'found', 'discovered', 'identified', 'observed',
                    'shows', 'demonstrates', 'indicates', 'suggests',
                    'causes', 'leads', 'results', 'produces',
                    'percentage', 'percent', 'number', 'amount',
                    'size', 'diameter', 'length', 'width', 'height'
                ]
                
                is_testable = any(keyword in sentence.lower() for keyword in testable_keywords)
                
                fallback_data.append({
                    "sentence_number": sentence_number,
                    "original_statement": sentence,
                    "is_testable": is_testable,
                    "optimized_assertion": sentence[:100] + "..." if len(sentence) > 100 else sentence
                })
                sentence_number += 1
        
        return fallback_data[:50]  # Limit to first 50 sentences to avoid overwhelming
    
    def find_relevant_documents(self, assertions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Find relevant documents for each assertion using Tavily search"""
        logger.info(f"Finding relevant documents for {len(assertions)} assertions")
        
        results = []
        
        for index, assertion in enumerate(assertions):
            logger.info(f"Processing {index+1}/{len(assertions)}: {assertion['Original Statement'][:100]}...")
            
            result_obj = {
                "Statement": assertion['Original Statement'],
                "Assertion": assertion['Assertion'],
                "Search Query": assertion['Original Statement'],
                "Summary (Tavily Answer)": "",
                "Relevant Docs": "",
                "Raw Content": ""
            }
            
            try:
                search_response = self.tavily_client.search(
                    query=assertion['Original Statement'],
                    search_depth="advanced",
                    include_domains=self.config.search_domains,
                    max_results=2,
                    include_raw_content=False,
                    include_answer=True
                )
                
                if search_response and "results" in search_response and search_response["results"]:
                    if "answer" in search_response:
                        result_obj["Summary (Tavily Answer)"] = search_response["answer"]
                    else:
                        result_obj["Summary (Tavily Answer)"] = "No answer provided by Tavily"
                    
                    formatted_results = self._format_documents(search_response["results"])
                    result_obj["Relevant Docs"] = formatted_results
                    
                    urls = [doc.get('url') for doc in search_response["results"] if doc.get('url')]
                    
                    if urls:
                        extracted_contents = self._extract_content_from_urls(urls)
                        combined_content = "\n\n---\n\n".join(extracted_contents)
                        result_obj["Raw Content"] = combined_content
                    else:
                        result_obj["Raw Content"] = "No URLs found to extract content from"
                else:
                    result_obj["Summary (Tavily Answer)"] = "No answer available"
                    result_obj["Relevant Docs"] = "No relevant documents found"
                    result_obj["Raw Content"] = "No content available"
                    
            except Exception as e:
                logger.error(f"Error searching for query '{assertion['Original Statement'][:100]}...': {e}")
                result_obj["Summary (Tavily Answer)"] = f"Error: {str(e)}"
                result_obj["Relevant Docs"] = f"Error: {str(e)}"
                result_obj["Raw Content"] = f"Error: {str(e)}"
            
            results.append(result_obj)
            time.sleep(1)  # Rate limiting
        
        logger.info(f"Document search completed for {len(results)} assertions")
        return results
    
    def _format_documents(self, documents: List[Dict]) -> str:
        """Format documents into a string for storage"""
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            formatted_docs.append(f"{i}. {doc.get('title', 'No title')} - {doc.get('url', 'No URL')}")
        return "\n".join(formatted_docs)
    
    def _extract_content_from_urls(self, urls: List[str], max_urls=2) -> List[str]:
        """Extract content from URLs using Tavily's extract feature"""
        extracted_contents = []
        
        for url in urls[:max_urls]:
            try:
                logger.info(f"Extracting content from: {url}")
                extract_response = self.tavily_client.extract(
                    urls=[url],
                    include_raw_content=True,
                    include_images=False
                )
                
                if extract_response:
                    content = self._extract_content_from_response(extract_response)
                    extracted_contents.append(content)
                else:
                    extracted_contents.append("No response from extract API")
                    
            except Exception as e:
                logger.error(f"Error extracting content from {url}: {e}")
                extracted_contents.append(f"Error extracting content: {str(e)}")
            
            time.sleep(1)  # Rate limiting
        
        return extracted_contents
    
    def _extract_content_from_response(self, response: Dict) -> str:
        """Extract content from Tavily extract response"""
        if "content" in response:
            return response["content"]
        elif "results" in response:
            results = response["results"]
            if results:
                first_result = results[0]
                for key in ["content", "raw_content", "text", "body"]:
                    if key in first_result:
                        return first_result[key]
        elif "raw_content" in response:
            return response["raw_content"]
        
        return "No content found in response"
    
    def build_knowledge_base(self, documents_data: List[Dict[str, str]]) -> str:
        """Build a knowledge base from extracted document content"""
        logger.info("Building knowledge base from document content")
        
        cleaned_blocks = []
        processed_items = 0
        
        for i, item in enumerate(documents_data):
            raw_content = item.get('Raw Content', '')
            if raw_content and raw_content not in ["No content available", "No URLs found to extract content from"]:
                cleaned = self._clean_content(raw_content)
                if cleaned:
                    cleaned_blocks.append(cleaned)
                    processed_items += 1
                    logger.info(f"Processed item {i+1}: {len(cleaned)} characters")
        
        if cleaned_blocks:
            final_text = "\n\n".join(cleaned_blocks)
            logger.info(f"Knowledge base built with {processed_items} items, {len(final_text)} characters")
            return final_text
        else:
            logger.warning("No content to build knowledge base from")
            return ""
    
    def _clean_content(self, content: str) -> str:
        """Clean content by removing unnecessary elements"""
        if not content or content in ["No content found in response", "No content found in extracted result"]:
            return ""
        
        # Remove various UI elements, navigation, and formatting
        patterns_to_remove = [
            r'!\[Image[^\]]*\]\([^)]*\)',  # Image references
            r'https?://[^\s\)]+',  # URLs
            r'www\.[^\s\)]+',  # Web addresses
            r'<[^>]+>',  # HTML tags
            r'Skip to main content',  # Navigation elements
            r'NCBI Homepage.*?MyNCBI Homepage.*?Main Content.*?Main Navigation',
            r'\[Log in\].*?Log out.*?',  # Login sections
            r'References.*?',  # References section
            r'Copyright.*?Bookshelf ID:.*?',  # Footer elements
            r'View on publisher site.*?Download PDF.*?Add to Collections.*?Cite.*?Permalink.*?',
            r'Back to Top.*?',  # Navigation
            r'Follow NCBI.*?',  # Social media
            r'Add to Collections.*?',  # Collection management
            r'Cite.*?',  # Citation elements
        ]
        
        for pattern in patterns_to_remove:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove markdown links but keep the text
        content = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', content)
        
        # Remove extra whitespace and newlines
        content = re.sub(r'\n+', '\n', content)
        content = re.sub(r' +', ' ', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r'^\s*$\n', '', content, flags=re.MULTILINE)
        
        return content.strip()

    # ------- NEW helpers to build metadata-aware KB blocks -------
    def _parse_relevant_docs_entries(self, relevant_docs_str: str) -> List[Dict[str, str]]:
        """
        Parse the human-readable 'Relevant Docs' string into [{'title', 'url'}, ...].
        Expected line format: '1. Title - https://...'
        """
        entries = []
        if not relevant_docs_str:
            return entries
        for line in relevant_docs_str.splitlines():
            line = line.strip()
            if not line:
                continue
            # remove leading enumeration like "1. " if present
            line = re.sub(r'^\d+\.\s*', '', line)
            if ' - ' in line:
                title, url = line.rsplit(' - ', 1)
                title, url = title.strip(), url.strip()
                if url.startswith('http'):
                    entries.append({"title": title or "Untitled", "url": url})
        return entries

    def _build_kb_blocks_from_documents(self, documents_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Produce structured KB blocks with metadata:
        [{'text': cleaned_text, 'source_title': str, 'source_url': str}, ...]
        Uses the same cleaned content your KB uses, but preserves URL/title per segment.
        """
        blocks = []
        SEP = "\n\n---\n\n"  # used when you joined extracted contents
        for item in documents_data:
            relevant_docs = self._parse_relevant_docs_entries(item.get("Relevant Docs", ""))
            raw = item.get("Raw Content", "") or ""
            if not raw or raw in ["No content available", "No URLs found to extract content from"]:
                continue

            segments = raw.split(SEP) if SEP in raw else [raw]
            # align only up to the min length to stay safe
            for idx in range(min(len(segments), len(relevant_docs))):
                seg = segments[idx]
                meta = relevant_docs[idx]
                cleaned = self._clean_content(seg)
                if cleaned:
                    blocks.append({
                        "text": cleaned,
                        "source_title": meta.get("title"),
                        "source_url": meta.get("url"),
                    })
        return blocks
    # ------------------------------------------------------------

    def fact_check_statements(self, documents_data: List[Dict[str, str]], knowledge_base: str) -> List[Dict[str, Any]]:
        """Fact check statements using RAG-enhanced Gemini analysis (JSON output + metadata-aware RAG)."""
        logger.info(f"Fact checking {len(documents_data)} statements")

        # 1) Try loading an existing index
        loaded = self.rag_system.load_index()

        # 2) If not available, build from structured blocks with metadata (preferred)
        if not loaded:
            kb_blocks = self._build_kb_blocks_from_documents(documents_data)
            if kb_blocks:
                self.rag_system.build_index_from_blocks(kb_blocks)
                self.rag_system.save_index()
            else:
                # fallback to legacy: string KB
                self.rag_system.build_index(knowledge_base)
                self.rag_system.save_index()

        results = []
        total_batches = (len(documents_data) + self.config.batch_size - 1) // self.config.batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(documents_data))
            batch_data = documents_data[start_idx:end_idx]
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} (statements {start_idx + 1}-{end_idx})")

            batch_results = self._fact_check_batch(batch_data)
            results.extend(batch_results)

            # Summary
            batch_verdicts = [r.get('Final Verdict', 'Flagged for Review') for r in batch_results]
            correct_count = batch_verdicts.count('Correct')
            incorrect_count = batch_verdicts.count('Incorrect')
            review_count = batch_verdicts.count('Flagged for Review')
            logger.info(f"Batch {batch_idx + 1} results: {correct_count} Correct, {incorrect_count} Incorrect, {review_count} Flagged for Review")

        logger.info(f"Fact checking completed for {len(results)} statements")
        return results
    
    def _fact_check_batch(self, batch_data: List[Dict]) -> List[Dict[str, Any]]:
        """Fact check a batch of statements using RAG-enhanced analysis"""
        try:
            # Create batch prompt
            prompt = self._create_batch_fact_checking_prompt(batch_data)
            
            # Generate response
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8000,
                    temperature=0.1,
                )
            )
            
            # Parse the batch response
            return self._parse_batch_response(response.text, batch_data)
            
        except Exception as e:
            logger.error(f"Error in batch fact checking: {e}")
            # Return default results for all statements in the batch
            return [
                {
                    "Statement": item.get("Statement", ""),
                    "Assertion": item.get("Assertion", ""),
                    "Summary (Tavily Answer)": item.get("Summary (Tavily Answer)", ""),
                    "Relevant Docs": item.get("Relevant Docs", ""),
                    "Final Verdict": "Flagged for Review",
                    "Full Analysis": f"Error occurred during batch fact checking: {str(e)}"
                }
                for item in batch_data
            ]
    
    def _create_batch_fact_checking_prompt(self, batch_data: List[Dict]) -> str:
        """Create a batch prompt for fact checking multiple statements (JSON-only output, includes citations)."""
        blocks = []
        for i, statement_data in enumerate(batch_data, 1):
            statement = statement_data.get("Statement", "")
            assertion = statement_data.get("Assertion", "")
            tavily_summary = statement_data.get("Summary (Tavily Answer)", "")
            relevant_docs = statement_data.get("Relevant Docs", "")

            # Use metadata-aware retrieval when available
            query = f"{statement} {assertion}".strip()
            rel = self.rag_system.retrieve_relevant_chunks_with_meta(query, top_k=3)
            if not rel:
                # fallback to legacy texts only
                rel_texts = self.rag_system.retrieve_relevant_chunks(query, top_k=2)
                rel = [{"text": t, "source_title": None, "source_url": None} for t in rel_texts] or [{"text": "No relevant information found in knowledge base.", "source_title": None, "source_url": None}]

            # pretty format for the prompt
            kb_lines = []
            for j, rc in enumerate(rel, 1):
                src = rc.get("source_url") or ""
                ttl = rc.get("source_title") or ""
                head = f"Chunk {j}"
                if ttl or src:
                    head += f" — {ttl} {f'({src})' if src else ''}"
                kb_lines.append(f"{head}:\n{rc.get('text','')}\n")

            kb_text = '\n'.join(kb_lines)
            block = f"""
<ITEM index="{i}">
STATEMENT: {statement}
ASSERTION: {assertion}
TAVILY_SUMMARY: {tavily_summary}
RELEVANT_DOCS_LIST:
{relevant_docs}

KNOWLEDGE_BASE_EXCERPTS:
{kb_text}
</ITEM>
"""
            blocks.append(block)

        instructions = """
You are a Scientific Fact Checking Agent with expertise in cancer biology and molecular biology.

TASK:
For EACH <ITEM>, evaluate the ASSERTION against the provided evidence (TAVILY_SUMMARY, RELEVANT_DOCS_LIST, KNOWLEDGE_BASE_EXCERPTS).

OUTPUT:
Return a SINGLE valid JSON array. Each element MUST be:
{
  "index": <int>,  // the <ITEM> index
  "final_verdict": "Correct" | "Incorrect" | "Flagged for Review",
  "reasoning": "<brief but specific justification>",
  "citations": ["<URL or short descriptor>", ...]  // include any URLs present in the knowledge excerpts or relevant docs
}

REQUIREMENTS:
- Output ONLY JSON. No markdown fences, no extra text.
- Keep "final_verdict" to exactly one of: "Correct", "Incorrect", "Flagged for Review".
- Prefer citations that correspond to the KNOWLEDGE_BASE_EXCERPTS (use their URLs if present).
"""
        return instructions + "\n\n" + "\n".join(blocks)
    
    def _parse_batch_response(self, response_text: str, batch_data: List[Dict]) -> List[Dict[str, Any]]:
        """Parse strict JSON batch response; robust to stray code fences."""
        # try to extract a JSON array
        text = response_text.strip()
        # strip code fences if the model added them
        m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if m:
            text = m.group(1).strip()

        # as a fallback, find the first [ ... ] block
        if not (text.startswith('[') and text.endswith(']')):
            m2 = re.search(r'\[\s*\{[\s\S]*\}\s*\]', text)
            if m2:
                text = m2.group(0)

        try:
            arr = json.loads(text)
            if not isinstance(arr, list):
                raise ValueError("Top-level JSON is not a list")
        except Exception as e:
            logger.error(f"Failed to parse JSON batch response: {e}")
            logger.error(text[:1000])
            # graceful fallback: mark all as review
            out = []
            for item in batch_data:
                out.append({
                    "Statement": item.get("Statement", ""),
                    "Assertion": item.get("Assertion", ""),
                    "Summary (Tavily Answer)": item.get("Summary (Tavily Answer)", ""),
                    "Relevant Docs": item.get("Relevant Docs", ""),
                    "Final Verdict": "Flagged for Review",
                    "Full Analysis": "Failed to parse JSON batch response."
                })
            return out

        # Map by index for safety
        idx_map = {int(obj.get("index")): obj for obj in arr if isinstance(obj, dict) and "index" in obj}
        results = []
        for i, item in enumerate(batch_data, 1):
            entry = idx_map.get(i, {})
            verdict = entry.get("final_verdict", "Flagged for Review")
            reasoning = entry.get("reasoning", "")
            citations = entry.get("citations", [])

            results.append({
                "Statement": item.get("Statement", ""),
                "Assertion": item.get("Assertion", ""),
                "Summary (Tavily Answer)": item.get("Summary (Tavily Answer)", ""),
                "Relevant Docs": item.get("Relevant Docs", ""),
                "Final Verdict": verdict if verdict in ("Correct", "Incorrect", "Flagged for Review") else "Flagged for Review",
                "Full Analysis": reasoning.strip(),
                "Citations": citations if isinstance(citations, list) else []
            })
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_dir: str = "output"):
        """Save results to CSV and JSON files"""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True)
        
        # Create DataFrame for CSV output
        output_df = pd.DataFrame(results)
        
        # Save to CSV
        csv_path = Path(output_dir) / "fact_checking_results.csv"
        output_df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to CSV: {csv_path}")
        
        # Save detailed results to JSON
        json_path = Path(output_dir) / "detailed_fact_checking_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed results saved to JSON: {json_path}")
        
        # Print summary
        verdict_counts = output_df['Final Verdict'].value_counts()
        logger.info("\nFinal Summary of Results:")
        for verdict, count in verdict_counts.items():
            logger.info(f"{verdict}: {count}")
        
        return csv_path, json_path

def load_config() -> Config:
    """Load configuration from environment variables"""
    google_api_key = os.getenv('GOOGLE_API_KEY')
    tavily_api_key = os.getenv('TAVILY_API_KEY')
    
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")
    
    search_domains = os.getenv('SEARCH_DOMAINS', 'ncbi.nlm.nih.gov,pubmed.ncbi.nlm.nih.gov').split(',')
    # trim whitespace on domains
    search_domains = [d.strip() for d in search_domains if d.strip()]
    batch_size = int(os.getenv('BATCH_SIZE', '5'))
    chunk_size = int(os.getenv('CHUNK_SIZE', '500'))
    
    return Config(
        google_api_key=google_api_key,
        tavily_api_key=tavily_api_key,
        search_domains=search_domains,
        batch_size=batch_size,
        chunk_size=chunk_size
    )

def main():
    """Main function to run the complete scientific fact checking pipeline"""
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize fact checker
        fact_checker = ScientificFactChecker(config)
        logger.info("Scientific fact checker initialized")
        
        # Step 1: Extract assertions from chapter
        chapter_path = "Chapters\\Chapter 02_ Introduction to Cancer_ A Disease of Deregulation.md"
        assertions = fact_checker.extract_assertions(chapter_path)
        
        if not assertions:
            logger.error("No assertions extracted. Exiting.")
            return
        
        # Save assertions to intermediate file
        with open("assertions_list.json", 'w', encoding='utf-8') as f:
            json.dump(assertions, f, indent=2, ensure_ascii=False)
        logger.info("Assertions saved to assertions_list.json")
        
        # Step 2: Find relevant documents
        documents_data = fact_checker.find_relevant_documents(assertions)
        
        # Save documents data to intermediate file
        with open("relevant_documents.json", 'w', encoding='utf-8') as f:
            json.dump(documents_data, f, indent=2, ensure_ascii=False)
        logger.info("Relevant documents saved to relevant_documents.json")
        
        # Step 3: Build knowledge base
        knowledge_base = fact_checker.build_knowledge_base(documents_data)
        
        # Save knowledge base
        with open("knowledge_base.txt", 'w', encoding='utf-8') as f:
            f.write(knowledge_base)
        logger.info("Knowledge base saved to knowledge_base.txt")
        
        # Step 4: Fact check statements
        results = fact_checker.fact_check_statements(documents_data, knowledge_base)
        
        # Step 5: Save results
        csv_path, json_path = fact_checker.save_results(results)
        
        logger.info("Scientific fact checking pipeline completed successfully!")
        logger.info(f"Results available at: {csv_path} and {json_path}")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# Scientific Fact Checking System
# A comprehensive tool for extracting, validating, and fact-checking scientific statements
# using Google Gemini AI and Tavily search integration.

# Author: Scientific Fact Checking Team
# Version: 1.0.0
# """

# import os
# import json
# import re
# import time
# import pickle
# import logging
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# from dataclasses import dataclass
# from datetime import datetime

# import google.generativeai as genai
# import pandas as pd
# import numpy as np
# from tavily import TavilyClient
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('fact_checker.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# @dataclass
# class Config:
#     """Configuration class for the fact checking system"""
#     google_api_key: str
#     tavily_api_key: str
#     search_domains: List[str]
#     batch_size: int
#     chunk_size: int
#     overlap: int = 50

# class SimpleRAGSystem:
#     """Simple RAG system using TF-IDF for document retrieval"""
    
#     def __init__(self, chunk_size=500, overlap=50):
#         self.chunk_size = chunk_size
#         self.overlap = overlap
#         self.chunks = []
#         self.vectorizer = TfidfVectorizer(
#             max_features=5000,
#             stop_words='english',
#             ngram_range=(1, 2)
#         )
#         self.chunk_vectors = None
#         self.is_fitted = False
        
#     def chunk_text(self, text: str) -> List[str]:
#         """Split text into overlapping chunks"""
#         chunks = []
#         words = text.split()
        
#         for i in range(0, len(words), self.chunk_size - self.overlap):
#             chunk_words = words[i:i + self.chunk_size]
#             chunk = ' '.join(chunk_words)
#             if len(chunk.strip()) > 50:  # Only keep chunks with meaningful content
#                 chunks.append(chunk.strip())
                
#         return chunks
    
#     def build_index(self, knowledge_base: str):
#         """Build the RAG index from knowledge base"""
#         logger.info("Building RAG index...")
        
#         # Chunk the knowledge base
#         self.chunks = self.chunk_text(knowledge_base)
#         logger.info(f"Created {len(self.chunks)} chunks from knowledge base")
        
#         # Create TF-IDF vectors
#         self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
#         self.is_fitted = True
        
#         logger.info("RAG index built successfully!")
        
#     def retrieve_relevant_chunks(self, query: str, top_k=3) -> List[str]:
#         """Retrieve top-k most relevant chunks for a query"""
#         if not self.is_fitted:
#             return []
            
#         # Vectorize the query
#         query_vector = self.vectorizer.transform([query])
        
#         # Calculate similarities
#         similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
        
#         # Get top-k indices
#         top_indices = np.argsort(similarities)[-top_k:][::-1]
        
#         # Return relevant chunks
#         relevant_chunks = [self.chunks[i] for i in top_indices if similarities[i] > 0.1]
        
#         return relevant_chunks
    
#     def save_index(self, filepath='rag_index.pkl'):
#         """Save the RAG index to disk"""
#         index_data = {
#             'chunks': self.chunks,
#             'vectorizer': self.vectorizer,
#             'chunk_vectors': self.chunk_vectors,
#             'is_fitted': self.is_fitted,
#             'chunk_size': self.chunk_size,
#             'overlap': self.overlap
#         }
        
#         with open(filepath, 'wb') as f:
#             pickle.dump(index_data, f)
#         logger.info(f"RAG index saved to {filepath}")
    
#     def load_index(self, filepath='rag_index.pkl'):
#         """Load the RAG index from disk"""
#         try:
#             with open(filepath, 'rb') as f:
#                 index_data = pickle.load(f)
                
#             self.chunks = index_data['chunks']
#             self.vectorizer = index_data['vectorizer']
#             self.chunk_vectors = index_data['chunk_vectors']
#             self.is_fitted = index_data['is_fitted']
#             self.chunk_size = index_data['chunk_size']
#             self.overlap = index_data['overlap']
            
#             logger.info(f"RAG index loaded from {filepath}")
#             return True
#         except FileNotFoundError:
#             logger.info(f"No existing index found at {filepath}")
#             return False

# class ScientificFactChecker:
#     """Main class for scientific fact checking system"""
    
#     def __init__(self, config: Config):
#         self.config = config
#         self.rag_system = SimpleRAGSystem(
#             chunk_size=config.chunk_size,
#             overlap=config.overlap
#         )
        
#         # Configure APIs
#         genai.configure(api_key=config.google_api_key)
#         self.tavily_client = TavilyClient(api_key=config.tavily_api_key)
#         self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        
#     def extract_assertions(self, chapter_path: str) -> List[Dict[str, str]]:
#         """Extract testable assertions from a markdown chapter"""
#         logger.info(f"Extracting assertions from: {chapter_path}")
        
#         try:
#             with open(chapter_path, "r", encoding="utf-8") as file:
#                 chapter_content = file.read()
#         except FileNotFoundError:
#             logger.error(f"File not found: {chapter_path}")
#             return []
        
#         # Get chapter name from filename
#         chapter_name = Path(chapter_path).stem.replace("_", " ")
        
#         # Create extraction prompt
#         prompt = self._create_extraction_prompt(chapter_name, chapter_content)
        
#         try:
#             response = self.gemini_model.generate_content(prompt)
#             extracted_data = self._parse_extraction_response(response.text)
            
#             # Filter and format testable assertions
#             testable_assertions = [
#                 {
#                     "Original Statement": item["original_statement"],
#                     "Assertion": item["optimized_assertion"]
#                 }
#                 for item in extracted_data if item.get("is_testable", False)
#             ]
            
#             logger.info(f"Extracted {len(testable_assertions)} testable assertions")
#             return testable_assertions
            
#         except Exception as e:
#             logger.error(f"Error extracting assertions: {e}")
#             return []
    
#     def _create_extraction_prompt(self, chapter_name: str, content: str) -> str:
#         """Create prompt for assertion extraction"""
#         return f"""
#         # SCIENTIFIC ASSERTION EXTRACTION PROMPT

#         ## YOUR ROLE
#         You are a SCIENTIFIC FACT-CHECKER tasked with extracting ALL verifiable assertions from scientific textbook content for peer-reviewed validation.

#         ## PRIMARY OBJECTIVE
#         EXTRACT MAXIMUM NUMBER of scientifically testable assertions from the provided chapter content. Your goal is COMPREHENSIVE COVERAGE - every sentence must be analyzed and NO SENTENCE should be overlooked.

#         ## CRITICAL REQUIREMENTS
#         - **MANDATORY**: Process EVERY SINGLE SENTENCE in the text
#         - **MAXIMIZE EXTRACTION**: Identify as many testable assertions as possible
#         - **ZERO OMISSIONS**: Absolutely no sentence should be left unanalyzed

#         ## TESTABLE ASSERTION DEFINITION
#         A testable assertion is ANY statement that can be verified against peer-reviewed scientific sources:
#         - Specific factual claims about properties, mechanisms, or relationships
#         - Quantitative data or measurements
#         - Cause-and-effect relationships
#         - Scientific processes or phenomena descriptions
#         - Research findings or experimental results

#         **EXCLUDE ONLY**: Pure definitions, introductory phrases, or entirely subjective opinions

#         ## ASSERTION OPTIMIZATION FOR SEARCH
#         For each testable statement, create an OPTIMIZED ASSERTION that:
#         - Contains the key scientific concepts and terminology
#         - Uses specific, searchable scientific terms
#         - Focuses on the most verifiable claim within the statement
#         - Will yield the most relevant peer-reviewed documents when searched
#         - Is concise but comprehensive enough for accurate fact-checking

#         ## EXTRACTION STEPS
#         1. **READ**: Process the text sentence by sentence sequentially
#         2. **ANALYZE**: Determine if each sentence contains verifiable scientific content
#         3. **EXTRACT**: Pull out the complete sentence containing the assertion
#         4. **OPTIMIZE**: Create a search-optimized assertion that will find the most relevant documents

#         ## OUTPUT FORMAT
#         Return your results in the following JSON format:
#         ```json
#         [
#         {{
#             "sentence_number": 1,
#             "original_statement": "EXACT sentence from text",
#             "is_testable": true,
#             "optimized_assertion": "Search-optimized assertion for finding relevant documents"
#         }},
#         {{
#             "sentence_number": 2,
#             "original_statement": "EXACT sentence from text",
#             "is_testable": false,
#             "optimized_assertion": "Search-optimized assertion for finding relevant documents"
#         }}
#         ]
#         ```
        
#         **CRITICAL JSON REQUIREMENTS:**
#         - Use ONLY double quotes (") for strings, NEVER single quotes
#         - Use lowercase true/false for boolean values
#         - Ensure every object is separated by commas
#         - No trailing commas after the last object
#         - All property names must be in double quotes
#         - Escape any quotes within text content with backslash (\")
#         - Ensure the JSON is valid and parseable

#         ## EXECUTION COMMAND
#         **Chapter Title**: {chapter_name}

#         **Content to Analyze**:
#         {content}

#         BEGIN COMPREHENSIVE SENTENCE-BY-SENTENCE ANALYSIS NOW.
#         """
    
#     def _parse_extraction_response(self, response_text: str) -> List[Dict]:
#         """Parse the extraction response to extract JSON data"""
#         # Remove markdown code formatting if present
#         json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
#         if json_match:
#             json_text = json_match.group(1)
#         else:
#             # If no code blocks, try to extract what looks like JSON
#             json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', response_text)
#             if json_match:
#                 json_text = json_match.group(0)
#             else:
#                 json_text = response_text

#         # Clean and fix common JSON issues
#         json_text = self._clean_json_text(json_text)
        
#         try:
#             return json.loads(json_text)
#         except json.JSONDecodeError as e:
#             logger.error(f"Error parsing JSON: {e}")
#             logger.error(f"JSON text: {json_text[:500]}...")  # Log first 500 chars for debugging
            
#             # Try one more aggressive cleaning approach
#             try:
#                 # Remove all problematic quotes and rebuild
#                 aggressive_clean = re.sub(r'(?<!\\)"(?![,\]\}:\s])', '\\"', json_text)
#                 return json.loads(aggressive_clean)
#             except json.JSONDecodeError:
#                 pass
            
#             # Try to extract basic information as fallback
#             logger.info("Attempting fallback extraction...")
#             fallback_data = self._extract_fallback_data(response_text)
#             if fallback_data:
#                 logger.info(f"Fallback extraction successful: {len(fallback_data)} items")
#                 return fallback_data
            
#             return []
    
#     def _clean_json_text(self, json_text: str) -> str:
#         """Clean and fix common JSON formatting issues"""
#         if not json_text:
#             return "[]"
        
#         # Remove any leading/trailing whitespace
#         json_text = json_text.strip()
        
#         # Fix common issues with AI-generated JSON
#         # 1. Fix missing commas between objects in arrays
#         json_text = re.sub(r'}\s*\n\s*{', '},\n{', json_text)
        
#         # 2. Fix trailing commas in arrays
#         json_text = re.sub(r',\s*]', ']', json_text)
        
#         # 3. Fix trailing commas in objects
#         json_text = re.sub(r',\s*}', '}', json_text)
        
#         # 4. Fix single quotes to double quotes first
#         json_text = json_text.replace("'", '"')
        
#         # 5. Fix missing quotes around property names (be more careful)
#         json_text = re.sub(r'(\s*)([a-zA-Z_]\w*)(\s*):', r'\1"\2"\3:', json_text)
        
#         # 6. Simple fix for unescaped quotes - replace problematic quotes in content
#         # This is a basic fix that handles most common cases
#         lines = json_text.split('\n')
#         fixed_lines = []
#         for line in lines:
#             if '": "' in line and line.count('"') > 4:  # Line has a string value with potential issues
#                 # Find the string value part and escape internal quotes
#                 parts = line.split('": "', 1)
#                 if len(parts) == 2:
#                     key_part = parts[0] + '": "'
#                     value_part = parts[1]
#                     # Find where the string value ends
#                     if value_part.endswith('",') or value_part.endswith('"'):
#                         if value_part.endswith('",'):
#                             end_part = '",'
#                             value_content = value_part[:-2]
#                         else:
#                             end_part = '"'
#                             value_content = value_part[:-1]
#                         # Escape quotes in the content
#                         value_content = value_content.replace('"', '\\"')
#                         line = key_part + value_content + end_part
#             fixed_lines.append(line)
#         json_text = '\n'.join(fixed_lines)
        
#         # 7. Fix boolean values (true/false should be lowercase)
#         json_text = re.sub(r'\bTrue\b', 'true', json_text)
#         json_text = re.sub(r'\bFalse\b', 'false', json_text)
        
#         # 8. Fix null values
#         json_text = re.sub(r'\bNone\b', 'null', json_text)
        
#         # 9. Remove any comments or extra text
#         lines = json_text.split('\n')
#         cleaned_lines = []
#         for line in lines:
#             line = line.strip()
#             if line and not line.startswith('//') and not line.startswith('#'):
#                 cleaned_lines.append(line)
        
#         json_text = '\n'.join(cleaned_lines)
        
#         # 10. Ensure the text starts and ends with proper brackets
#         if not json_text.startswith('['):
#             json_text = '[' + json_text
#         if not json_text.endswith(']'):
#             json_text = json_text + ']'
        
#         return json_text
    
#     def _extract_fallback_data(self, response_text: str) -> List[Dict]:
#         """Fallback method to extract data when JSON parsing fails"""
#         fallback_data = []
        
#         # Look for patterns that indicate testable statements
#         # Pattern 1: Look for sentences that might be testable
#         sentences = re.split(r'[.!?]+', response_text)
        
#         sentence_number = 1
#         for sentence in sentences:
#             sentence = sentence.strip()
#             if len(sentence) > 20:  # Only consider meaningful sentences
#                 # Simple heuristics to determine if a sentence might be testable
#                 testable_keywords = [
#                     'is', 'are', 'was', 'were', 'has', 'have', 'had',
#                     'contains', 'consists', 'includes', 'comprises',
#                     'found', 'discovered', 'identified', 'observed',
#                     'shows', 'demonstrates', 'indicates', 'suggests',
#                     'causes', 'leads', 'results', 'produces',
#                     'percentage', 'percent', 'number', 'amount',
#                     'size', 'diameter', 'length', 'width', 'height'
#                 ]
                
#                 is_testable = any(keyword in sentence.lower() for keyword in testable_keywords)
                
#                 fallback_data.append({
#                     "sentence_number": sentence_number,
#                     "original_statement": sentence,
#                     "is_testable": is_testable,
#                     "optimized_assertion": sentence[:100] + "..." if len(sentence) > 100 else sentence
#                 })
#                 sentence_number += 1
        
#         return fallback_data[:50]  # Limit to first 50 sentences to avoid overwhelming
    
#     def find_relevant_documents(self, assertions: List[Dict[str, str]]) -> List[Dict[str, str]]:
#         """Find relevant documents for each assertion using Tavily search"""
#         logger.info(f"Finding relevant documents for {len(assertions)} assertions")
        
#         results = []
        
#         for index, assertion in enumerate(assertions):
#             logger.info(f"Processing {index+1}/{len(assertions)}: {assertion['Original Statement'][:100]}...")
            
#             result_obj = {
#                 "Statement": assertion['Original Statement'],
#                 "Assertion": assertion['Assertion'],
#                 "Search Query": assertion['Original Statement'],
#                 "Summary (Tavily Answer)": "",
#                 "Relevant Docs": "",
#                 "Raw Content": ""
#             }
            
#             try:
#                 search_response = self.tavily_client.search(
#                     query=assertion['Original Statement'],
#                     search_depth="advanced",
#                     include_domains=self.config.search_domains,
#                     max_results=2,
#                     include_raw_content=False,
#                     include_answer=True
#                 )
                
#                 if search_response and "results" in search_response and search_response["results"]:
#                     if "answer" in search_response:
#                         result_obj["Summary (Tavily Answer)"] = search_response["answer"]
#                     else:
#                         result_obj["Summary (Tavily Answer)"] = "No answer provided by Tavily"
                    
#                     formatted_results = self._format_documents(search_response["results"])
#                     result_obj["Relevant Docs"] = formatted_results
                    
#                     urls = [doc.get('url') for doc in search_response["results"] if doc.get('url')]
                    
#                     if urls:
#                         extracted_contents = self._extract_content_from_urls(urls)
#                         combined_content = "\n\n---\n\n".join(extracted_contents)
#                         result_obj["Raw Content"] = combined_content
#                     else:
#                         result_obj["Raw Content"] = "No URLs found to extract content from"
#                 else:
#                     result_obj["Summary (Tavily Answer)"] = "No answer available"
#                     result_obj["Relevant Docs"] = "No relevant documents found"
#                     result_obj["Raw Content"] = "No content available"
                    
#             except Exception as e:
#                 logger.error(f"Error searching for query '{assertion['Original Statement'][:100]}...': {e}")
#                 result_obj["Summary (Tavily Answer)"] = f"Error: {str(e)}"
#                 result_obj["Relevant Docs"] = f"Error: {str(e)}"
#                 result_obj["Raw Content"] = f"Error: {str(e)}"
            
#             results.append(result_obj)
#             time.sleep(1)  # Rate limiting
        
#         logger.info(f"Document search completed for {len(results)} assertions")
#         return results
    
#     def _format_documents(self, documents: List[Dict]) -> str:
#         """Format documents into a string for storage"""
#         formatted_docs = []
#         for i, doc in enumerate(documents, 1):
#             formatted_docs.append(f"{i}. {doc.get('title', 'No title')} - {doc.get('url', 'No URL')}")
#         return "\n".join(formatted_docs)
    
#     def _extract_content_from_urls(self, urls: List[str], max_urls=2) -> List[str]:
#         """Extract content from URLs using Tavily's extract feature"""
#         extracted_contents = []
        
#         for url in urls[:max_urls]:
#             try:
#                 logger.info(f"Extracting content from: {url}")
#                 extract_response = self.tavily_client.extract(
#                     urls=[url],
#                     include_raw_content=True,
#                     include_images=False
#                 )
                
#                 if extract_response:
#                     content = self._extract_content_from_response(extract_response)
#                     extracted_contents.append(content)
#                 else:
#                     extracted_contents.append("No response from extract API")
                    
#             except Exception as e:
#                 logger.error(f"Error extracting content from {url}: {e}")
#                 extracted_contents.append(f"Error extracting content: {str(e)}")
            
#             time.sleep(1)  # Rate limiting
        
#         return extracted_contents
    
#     def _extract_content_from_response(self, response: Dict) -> str:
#         """Extract content from Tavily extract response"""
#         if "content" in response:
#             return response["content"]
#         elif "results" in response:
#             results = response["results"]
#             if results:
#                 first_result = results[0]
#                 for key in ["content", "raw_content", "text", "body"]:
#                     if key in first_result:
#                         return first_result[key]
#         elif "raw_content" in response:
#             return response["raw_content"]
        
#         return "No content found in response"
    
#     def build_knowledge_base(self, documents_data: List[Dict[str, str]]) -> str:
#         """Build a knowledge base from extracted document content"""
#         logger.info("Building knowledge base from document content")
        
#         cleaned_blocks = []
#         processed_items = 0
        
#         for i, item in enumerate(documents_data):
#             raw_content = item.get('Raw Content', '')
#             if raw_content and raw_content not in ["No content available", "No URLs found to extract content from"]:
#                 cleaned = self._clean_content(raw_content)
#                 if cleaned:
#                     cleaned_blocks.append(cleaned)
#                     processed_items += 1
#                     logger.info(f"Processed item {i+1}: {len(cleaned)} characters")
        
#         if cleaned_blocks:
#             final_text = "\n\n".join(cleaned_blocks)
#             logger.info(f"Knowledge base built with {processed_items} items, {len(final_text)} characters")
#             return final_text
#         else:
#             logger.warning("No content to build knowledge base from")
#             return ""
    
#     def _clean_content(self, content: str) -> str:
#         """Clean content by removing unnecessary elements"""
#         if not content or content in ["No content found in response", "No content found in extracted result"]:
#             return ""
        
#         # Remove various UI elements, navigation, and formatting
#         patterns_to_remove = [
#             r'!\[Image[^\]]*\]\([^)]*\)',  # Image references
#             r'https?://[^\s\)]+',  # URLs
#             r'www\.[^\s\)]+',  # Web addresses
#             r'<[^>]+>',  # HTML tags
#             r'Skip to main content',  # Navigation elements
#             r'NCBI Homepage.*?MyNCBI Homepage.*?Main Content.*?Main Navigation',
#             r'\[Log in\].*?Log out.*?',  # Login sections
#             r'References.*?',  # References section
#             r'Copyright.*?Bookshelf ID:.*?',  # Footer elements
#             r'View on publisher site.*?Download PDF.*?Add to Collections.*?Cite.*?Permalink.*?',
#             r'Back to Top.*?',  # Navigation
#             r'Follow NCBI.*?',  # Social media
#             r'Add to Collections.*?',  # Collection management
#             r'Cite.*?',  # Citation elements
#         ]
        
#         for pattern in patterns_to_remove:
#             content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
#         # Remove markdown links but keep the text
#         content = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', content)
        
#         # Remove extra whitespace and newlines
#         content = re.sub(r'\n+', '\n', content)
#         content = re.sub(r' +', ' ', content)
#         content = re.sub(r'\n\s*\n', '\n\n', content)
#         content = re.sub(r'^\s*$\n', '', content, flags=re.MULTILINE)
        
#         return content.strip()
    
#     def fact_check_statements(self, documents_data: List[Dict[str, str]], knowledge_base: str) -> List[Dict[str, Any]]:
#         """Fact check statements using RAG-enhanced Gemini analysis"""
#         logger.info(f"Fact checking {len(documents_data)} statements")
        
#         # Initialize RAG system
#         if not self.rag_system.load_index():
#             self.rag_system.build_index(knowledge_base)
#             self.rag_system.save_index()
        
#         results = []
        
#         # Process statements in batches
#         total_batches = (len(documents_data) + self.config.batch_size - 1) // self.config.batch_size
        
#         for batch_idx in range(total_batches):
#             start_idx = batch_idx * self.config.batch_size
#             end_idx = min(start_idx + self.config.batch_size, len(documents_data))
#             batch_data = documents_data[start_idx:end_idx]
            
#             logger.info(f"Processing batch {batch_idx + 1}/{total_batches} (statements {start_idx + 1}-{end_idx})")
            
#             # Fact check the batch
#             batch_results = self._fact_check_batch(batch_data)
#             results.extend(batch_results)
            
#             # Print batch results summary
#             batch_verdicts = [result['Final Verdict'] for result in batch_results]
#             correct_count = batch_verdicts.count('Correct')
#             incorrect_count = batch_verdicts.count('Incorrect')
#             review_count = batch_verdicts.count('Flagged for Review')
            
#             logger.info(f"Batch {batch_idx + 1} results: {correct_count} Correct, {incorrect_count} Incorrect, {review_count} Flagged for Review")
        
#         logger.info(f"Fact checking completed for {len(results)} statements")
#         return results
    
#     def _fact_check_batch(self, batch_data: List[Dict]) -> List[Dict[str, Any]]:
#         """Fact check a batch of statements using RAG-enhanced analysis"""
#         try:
#             # Create batch prompt
#             prompt = self._create_batch_fact_checking_prompt(batch_data)
            
#             # Generate response
#             response = self.gemini_model.generate_content(
#                 prompt,
#                 generation_config=genai.types.GenerationConfig(
#                     max_output_tokens=8000,
#                     temperature=0.1,
#                 )
#             )
            
#             # Parse the batch response
#             return self._parse_batch_response(response.text, batch_data)
            
#         except Exception as e:
#             logger.error(f"Error in batch fact checking: {e}")
#             # Return default results for all statements in the batch
#             return [
#                 {
#                     "Statement": item.get("Statement", ""),
#                     "Assertion": item.get("Assertion", ""),
#                     "Summary (Tavily Answer)": item.get("Summary (Tavily Answer)", ""),
#                     "Relevant Docs": item.get("Relevant Docs", ""),
#                     "Final Verdict": "Flagged for Review",
#                     "Full Analysis": f"Error occurred during batch fact checking: {str(e)}"
#                 }
#                 for item in batch_data
#             ]
    
#     def _create_batch_fact_checking_prompt(self, batch_data: List[Dict]) -> str:
#         """Create a batch prompt for fact checking multiple statements"""
#         batch_statements = []
        
#         for i, statement_data in enumerate(batch_data, 1):
#             statement = statement_data.get("Statement", "")
#             assertion = statement_data.get("Assertion", "")
#             tavily_summary = statement_data.get("Summary (Tavily Answer)", "")
#             relevant_docs = statement_data.get("Relevant Docs", "")
            
#             # Get relevant chunks for this statement
#             query = f"{statement} {assertion}"
#             relevant_chunks = self.rag_system.retrieve_relevant_chunks(query, top_k=2)
            
#             if not relevant_chunks:
#                 relevant_chunks = ["No relevant information found in knowledge base."]
            
#             relevant_knowledge = "\n".join([f"  Chunk {j+1}: {chunk}" for j, chunk in enumerate(relevant_chunks)])
            
#             statement_block = f"""
# STATEMENT {i}:
# Statement: "{statement}"
# Assertion: "{assertion}"
# Tavily Answer Summary: "{tavily_summary}"
# Relevant Documents: {relevant_docs}

# Relevant Knowledge Base Excerpts:
# {relevant_knowledge}
# """
#             batch_statements.append(statement_block)
        
#         statements_text = "\n".join(batch_statements)
        
#         return f"""
# You are a Scientific Fact Checking Agent with expertise in cancer biology and molecular biology. Your role is to evaluate the accuracy of multiple scientific statements using authoritative sources and evidence.

# TASK:
# Fact check the following {len(batch_data)} statements using the provided evidence and knowledge base excerpts for each statement.

# INSTRUCTIONS:
# 1. Carefully analyze each statement against its provided evidence
# 2. Use the relevant knowledge chunks as your primary reference source for each statement
# 3. Consider the Tavily Answer summary as additional context for each statement
# 4. Evaluate each statement for factual accuracy, completeness, and scientific validity
# 5. Provide a final verdict for each statement based on the evidence

# STATEMENTS TO EVALUATE:
# {statements_text}

# EVALUATION CRITERIA:
# - Correct: The statement is factually accurate and supported by the evidence
# - Incorrect: The statement contains factual errors or contradicts the evidence
# - Flagged for Review: The statement requires additional verification or contains ambiguous/contradictory information

# OUTPUT FORMAT:
# For each statement, provide your analysis in the following structure:

# STATEMENT 1 ANALYSIS:
# [Detailed analysis of the statement's accuracy, including specific evidence from the knowledge base]

# STATEMENT 1 EVIDENCE ASSESSMENT:
# [Evaluation of the strength and relevance of the provided evidence]

# STATEMENT 1 FINAL VERDICT:
# [Choose one: Correct, Incorrect, or Flagged for Review]

# STATEMENT 1 REASONING:
# [Clear explanation for your verdict based on the evidence]

# [Repeat for all {len(batch_data)} statements]
# """
    
#     def _parse_batch_response(self, response_text: str, batch_data: List[Dict]) -> List[Dict[str, Any]]:
#         """Parse the batch response to extract results for multiple statements"""
#         results = []
        
#         for i, statement_data in enumerate(batch_data, 1):
#             statement = statement_data.get("Statement", "")
#             assertion = statement_data.get("Assertion", "")
#             tavily_summary = statement_data.get("Summary (Tavily Answer)", "")
#             relevant_docs = statement_data.get("Relevant Docs", "")
            
#             # Look for this statement's verdict in the response
#             final_verdict = "Flagged for Review"  # Default
            
#             # Try multiple patterns to find the verdict for this statement
#             patterns = [
#                 f"STATEMENT {i} FINAL VERDICT:",
#                 f"STATEMENT {i} VERDICT:",
#                 f"Statement {i} Final Verdict:",
#                 f"Statement {i} Verdict:"
#             ]
            
#             statement_analysis = ""
            
#             for pattern in patterns:
#                 if pattern in response_text:
#                     # Extract the section for this statement
#                     start_idx = response_text.find(pattern)
#                     end_patterns = [f"STATEMENT {i+1}", f"Statement {i+1}", "STATEMENT", "Statement"]
                    
#                     end_idx = len(response_text)
#                     for end_pattern in end_patterns:
#                         temp_idx = response_text.find(end_pattern, start_idx + len(pattern))
#                         if temp_idx != -1:
#                             end_idx = temp_idx
#                             break
                    
#                     statement_section = response_text[start_idx:end_idx]
#                     statement_analysis = statement_section
                    
#                     # Extract verdict from this section
#                     if "Correct" in statement_section and "Incorrect" not in statement_section:
#                         final_verdict = "Correct"
#                     elif "Incorrect" in statement_section and "Correct" not in statement_section:
#                         final_verdict = "Incorrect"
#                     elif "Flagged for Review" in statement_section:
#                         final_verdict = "Flagged for Review"
#                     break
            
#             # If we couldn't find specific statement analysis, use a portion of the full response
#             if not statement_analysis:
#                 statement_analysis = f"Analysis for statement {i} not clearly separated in batch response."
            
#             results.append({
#                 "Statement": statement,
#                 "Assertion": assertion,
#                 "Summary (Tavily Answer)": tavily_summary,
#                 "Relevant Docs": relevant_docs,
#                 "Final Verdict": final_verdict,
#                 "Full Analysis": statement_analysis.strip()
#             })
        
#         return results
    
#     def save_results(self, results: List[Dict[str, Any]], output_dir: str = "output"):
#         """Save results to CSV and JSON files"""
#         # Create output directory if it doesn't exist
#         Path(output_dir).mkdir(exist_ok=True)
        
#         # Create DataFrame for CSV output
#         output_df = pd.DataFrame(results)
        
#         # Save to CSV
#         csv_path = Path(output_dir) / "fact_checking_results.csv"
#         output_df.to_csv(csv_path, index=False)
#         logger.info(f"Results saved to CSV: {csv_path}")
        
#         # Save detailed results to JSON
#         json_path = Path(output_dir) / "detailed_fact_checking_results.json"
#         with open(json_path, 'w', encoding='utf-8') as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)
#         logger.info(f"Detailed results saved to JSON: {json_path}")
        
#         # Print summary
#         verdict_counts = output_df['Final Verdict'].value_counts()
#         logger.info("\nFinal Summary of Results:")
#         for verdict, count in verdict_counts.items():
#             logger.info(f"{verdict}: {count}")
        
#         return csv_path, json_path

# def load_config() -> Config:
#     """Load configuration from environment variables"""
#     google_api_key = os.getenv('GOOGLE_API_KEY')
#     tavily_api_key = os.getenv('TAVILY_API_KEY')
    
#     if not google_api_key:
#         raise ValueError("GOOGLE_API_KEY environment variable is required")
#     if not tavily_api_key:
#         raise ValueError("TAVILY_API_KEY environment variable is required")
    
#     search_domains = os.getenv('SEARCH_DOMAINS', 'ncbi.nlm.nih.gov,pubmed.ncbi.nlm.nih.gov').split(',')
#     batch_size = int(os.getenv('BATCH_SIZE', '5'))
#     chunk_size = int(os.getenv('CHUNK_SIZE', '500'))
    
#     return Config(
#         google_api_key=google_api_key,
#         tavily_api_key=tavily_api_key,
#         search_domains=search_domains,
#         batch_size=batch_size,
#         chunk_size=chunk_size
#     )

# def main():
#     """Main function to run the complete scientific fact checking pipeline"""
#     try:
#         # Load configuration
#         config = load_config()
#         logger.info("Configuration loaded successfully")
        
#         # Initialize fact checker
#         fact_checker = ScientificFactChecker(config)
#         logger.info("Scientific fact checker initialized")
        
#         # Step 1: Extract assertions from chapter
#         chapter_path = "Chapters\Chapter 02_ Introduction to Cancer_ A Disease of Deregulation.md"
#         assertions = fact_checker.extract_assertions(chapter_path)
        
#         if not assertions:
#             logger.error("No assertions extracted. Exiting.")
#             return
        
#         # Save assertions to intermediate file
#         with open("assertions_list.json", 'w', encoding='utf-8') as f:
#             json.dump(assertions, f, indent=2, ensure_ascii=False)
#         logger.info("Assertions saved to assertions_list.json")
        
#         # Step 2: Find relevant documents
#         documents_data = fact_checker.find_relevant_documents(assertions)
        
#         # Save documents data to intermediate file
#         with open("relevant_documents.json", 'w', encoding='utf-8') as f:
#             json.dump(documents_data, f, indent=2, ensure_ascii=False)
#         logger.info("Relevant documents saved to relevant_documents.json")
        
#         # Step 3: Build knowledge base
#         knowledge_base = fact_checker.build_knowledge_base(documents_data)
        
#         # Save knowledge base
#         with open("knowledge_base.txt", 'w', encoding='utf-8') as f:
#             f.write(knowledge_base)
#         logger.info("Knowledge base saved to knowledge_base.txt")
        
#         # Step 4: Fact check statements
#         results = fact_checker.fact_check_statements(documents_data, knowledge_base)
        
#         # Step 5: Save results
#         csv_path, json_path = fact_checker.save_results(results)
        
#         logger.info("Scientific fact checking pipeline completed successfully!")
#         logger.info(f"Results available at: {csv_path} and {json_path}")
        
#     except Exception as e:
#         logger.error(f"Error in main pipeline: {e}")
#         raise

# if __name__ == "__main__":
#     main()

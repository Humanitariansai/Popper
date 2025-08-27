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
    """Simple RAG system using TF-IDF for document retrieval"""
    
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.chunk_vectors = None
        self.is_fitted = False
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            if len(chunk.strip()) > 50:  # Only keep chunks with meaningful content
                chunks.append(chunk.strip())
                
        return chunks
    
    def build_index(self, knowledge_base: str):
        """Build the RAG index from knowledge base"""
        logger.info("Building RAG index...")
        
        # Chunk the knowledge base
        self.chunks = self.chunk_text(knowledge_base)
        logger.info(f"Created {len(self.chunks)} chunks from knowledge base")
        
        # Create TF-IDF vectors
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
        self.is_fitted = True
        
        logger.info("RAG index built successfully!")
        
    def retrieve_relevant_chunks(self, query: str, top_k=3) -> List[str]:
        """Retrieve top-k most relevant chunks for a query"""
        if not self.is_fitted:
            return []
            
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return relevant chunks
        relevant_chunks = [self.chunks[i] for i in top_indices if similarities[i] > 0.1]
        
        return relevant_chunks
    
    def save_index(self, filepath='rag_index.pkl'):
        """Save the RAG index to disk"""
        index_data = {
            'chunks': self.chunks,
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
                
            self.chunks = index_data['chunks']
            self.vectorizer = index_data['vectorizer']
            self.chunk_vectors = index_data['chunk_vectors']
            self.is_fitted = index_data['is_fitted']
            self.chunk_size = index_data['chunk_size']
            self.overlap = index_data['overlap']
            
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
                    "Assertion": item["optimized_assertion"],
                    "Searchable Query": item["search_query"]
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
        5. **QUERY**: Create focused search terms (3-7 words) targeting key scientific concepts

        ## OUTPUT FORMAT
        Return your results in the following JSON format:
        ```json
        [
        {{
            "sentence_number": 1,
            "original_statement": "EXACT sentence from text",
            "is_testable": true/false,
            "optimized_assertion": "Search-optimized assertion for finding relevant documents",
            "search_query": "focused scientific search terms"
        }},
        // More statements...
        ]
        ```

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

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            return []
    
    def find_relevant_documents(self, assertions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Find relevant documents for each assertion using Tavily search"""
        logger.info(f"Finding relevant documents for {len(assertions)} assertions")
        
        results = []
        
        for index, assertion in enumerate(assertions):
            logger.info(f"Processing {index+1}/{len(assertions)}: {assertion['Searchable Query']}")
            
            result_obj = {
                "Statement": assertion['Original Statement'],
                "Assertion": assertion['Assertion'],
                "Search Query": assertion['Searchable Query'],
                "Summary (Tavily Answer)": "",
                "Relevant Docs": "",
                "Raw Content": ""
            }
            
            try:
                search_response = self.tavily_client.search(
                    query=assertion['Searchable Query'],
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
                logger.error(f"Error searching for query '{assertion['Searchable Query']}': {e}")
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
    
    def fact_check_statements(self, documents_data: List[Dict[str, str]], knowledge_base: str) -> List[Dict[str, Any]]:
        """Fact check statements using RAG-enhanced Gemini analysis"""
        logger.info(f"Fact checking {len(documents_data)} statements")
        
        # Initialize RAG system
        if not self.rag_system.load_index():
            self.rag_system.build_index(knowledge_base)
            self.rag_system.save_index()
        
        results = []
        
        # Process statements in batches
        total_batches = (len(documents_data) + self.config.batch_size - 1) // self.config.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(documents_data))
            batch_data = documents_data[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} (statements {start_idx + 1}-{end_idx})")
            
            # Fact check the batch
            batch_results = self._fact_check_batch(batch_data)
            results.extend(batch_results)
            
            # Print batch results summary
            batch_verdicts = [result['Final Verdict'] for result in batch_results]
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
        """Create a batch prompt for fact checking multiple statements"""
        batch_statements = []
        
        for i, statement_data in enumerate(batch_data, 1):
            statement = statement_data.get("Statement", "")
            assertion = statement_data.get("Assertion", "")
            tavily_summary = statement_data.get("Summary (Tavily Answer)", "")
            relevant_docs = statement_data.get("Relevant Docs", "")
            
            # Get relevant chunks for this statement
            query = f"{statement} {assertion}"
            relevant_chunks = self.rag_system.retrieve_relevant_chunks(query, top_k=2)
            
            if not relevant_chunks:
                relevant_chunks = ["No relevant information found in knowledge base."]
            
            relevant_knowledge = "\n".join([f"  Chunk {j+1}: {chunk}" for j, chunk in enumerate(relevant_chunks)])
            
            statement_block = f"""
STATEMENT {i}:
Statement: "{statement}"
Assertion: "{assertion}"
Tavily Answer Summary: "{tavily_summary}"
Relevant Documents: {relevant_docs}

Relevant Knowledge Base Excerpts:
{relevant_knowledge}
"""
            batch_statements.append(statement_block)
        
        statements_text = "\n".join(batch_statements)
        
        return f"""
You are a Scientific Fact Checking Agent with expertise in cancer biology and molecular biology. Your role is to evaluate the accuracy of multiple scientific statements using authoritative sources and evidence.

TASK:
Fact check the following {len(batch_data)} statements using the provided evidence and knowledge base excerpts for each statement.

INSTRUCTIONS:
1. Carefully analyze each statement against its provided evidence
2. Use the relevant knowledge chunks as your primary reference source for each statement
3. Consider the Tavily Answer summary as additional context for each statement
4. Evaluate each statement for factual accuracy, completeness, and scientific validity
5. Provide a final verdict for each statement based on the evidence

STATEMENTS TO EVALUATE:
{statements_text}

EVALUATION CRITERIA:
- Correct: The statement is factually accurate and supported by the evidence
- Incorrect: The statement contains factual errors or contradicts the evidence
- Flagged for Review: The statement requires additional verification or contains ambiguous/contradictory information

OUTPUT FORMAT:
For each statement, provide your analysis in the following structure:

STATEMENT 1 ANALYSIS:
[Detailed analysis of the statement's accuracy, including specific evidence from the knowledge base]

STATEMENT 1 EVIDENCE ASSESSMENT:
[Evaluation of the strength and relevance of the provided evidence]

STATEMENT 1 FINAL VERDICT:
[Choose one: Correct, Incorrect, or Flagged for Review]

STATEMENT 1 REASONING:
[Clear explanation for your verdict based on the evidence]

[Repeat for all {len(batch_data)} statements]
"""
    
    def _parse_batch_response(self, response_text: str, batch_data: List[Dict]) -> List[Dict[str, Any]]:
        """Parse the batch response to extract results for multiple statements"""
        results = []
        
        for i, statement_data in enumerate(batch_data, 1):
            statement = statement_data.get("Statement", "")
            assertion = statement_data.get("Assertion", "")
            tavily_summary = statement_data.get("Summary (Tavily Answer)", "")
            relevant_docs = statement_data.get("Relevant Docs", "")
            
            # Look for this statement's verdict in the response
            final_verdict = "Flagged for Review"  # Default
            
            # Try multiple patterns to find the verdict for this statement
            patterns = [
                f"STATEMENT {i} FINAL VERDICT:",
                f"STATEMENT {i} VERDICT:",
                f"Statement {i} Final Verdict:",
                f"Statement {i} Verdict:"
            ]
            
            statement_analysis = ""
            
            for pattern in patterns:
                if pattern in response_text:
                    # Extract the section for this statement
                    start_idx = response_text.find(pattern)
                    end_patterns = [f"STATEMENT {i+1}", f"Statement {i+1}", "STATEMENT", "Statement"]
                    
                    end_idx = len(response_text)
                    for end_pattern in end_patterns:
                        temp_idx = response_text.find(end_pattern, start_idx + len(pattern))
                        if temp_idx != -1:
                            end_idx = temp_idx
                            break
                    
                    statement_section = response_text[start_idx:end_idx]
                    statement_analysis = statement_section
                    
                    # Extract verdict from this section
                    if "Correct" in statement_section and "Incorrect" not in statement_section:
                        final_verdict = "Correct"
                    elif "Incorrect" in statement_section and "Correct" not in statement_section:
                        final_verdict = "Incorrect"
                    elif "Flagged for Review" in statement_section:
                        final_verdict = "Flagged for Review"
                    break
            
            # If we couldn't find specific statement analysis, use a portion of the full response
            if not statement_analysis:
                statement_analysis = f"Analysis for statement {i} not clearly separated in batch response."
            
            results.append({
                "Statement": statement,
                "Assertion": assertion,
                "Summary (Tavily Answer)": tavily_summary,
                "Relevant Docs": relevant_docs,
                "Final Verdict": final_verdict,
                "Full Analysis": statement_analysis.strip()
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
    batch_size = int(os.getenv('BATCH_SIZE', '10'))
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
        chapter_path = "Chapters\Chapter 02_ Introduction to Cancer_ A Disease of Deregulation.md"
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

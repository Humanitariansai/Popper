import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv         
import google.generativeai as genai
from textwrap import dedent

def configure_sdk():
    """
    Load environment variables and configure the Generative AI SDK.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)

def get_assertion_prompt(input_text: str) -> str:
    """
    Return a prompt to extract assertions from the input text.
    """
    base_prompt = dedent("""
        You are an expert AI trained to extract and classify scientific assertions.

        Task:
        Analyze the [INPUT TEXT] and identify all verifiable factual statements (Assertions). Exclude:

        Opinions (e.g., “This is the most important discovery…”)
        Questions (e.g., “What is the function of…?”)
        Hypotheses (e.g., “We hypothesize that…”)
        Procedures (e.g., “First, mix the solution…”)

        Each Assertion must be classified into one of the following six subtypes:

        Assertion Subtypes:
        Definitional/Taxonomic
        Defines or categorizes entities
        Keywords: is, refers to, defined as, known as, classified as, belongs to, includes, consists of
        Example: "Apoptosis is a form of programmed cell death."

        Causal/Mechanistic
        Describes cause-effect relationships or mechanisms
        Keywords: causes, leads to, induces, results in, activates, stimulates, suppresses, mediates, modulates, is triggered by, depends on, is required for
        Example: "IL-6 activates the JAK-STAT3 signaling pathway."

        Quantitative/Statistical
        Presents numeric values or measurements
        Keywords: %, rate, ratio, X per Y, prevalence, concentration, mean, median, range, standard deviation
        Example: "The prevalence of type 2 diabetes in adults over 50 is 14.2%."

        Observational/Correlational
        Reports observed associations without implying causation
        Keywords: associated with, correlated with, linked to, more likely to, observed in, found in, coincides with, tends to
        Example: "Vitamin D deficiency is linked to depressive symptoms."

        Experimental/Interventional
        Reports results from studies or interventions
        Keywords: we found that, the study showed, in treated X, we observed, administration of, participants receiving, resulted in, demonstrated
        Example: "Treatment with metformin reduced HbA1c levels in diabetic patients."

        Comparative
        Compares entities by effectiveness, structure, frequency, etc.
        Keywords: higher than, lower than, more effective than, compared to, superior to, less frequent, similar to, no significant difference
        Example: "Radiation therapy is more effective than chemotherapy in early-stage prostate cancer."

        Output Format:
        Provide a JSON array of extracted assertions. No extra explanation.
        Each object must include:

        id: Integer (starting from 1)
        assertion_subtype: One of the 6 types above
        statement_text: Verbatim extracted assertion

        Example Input:
        "The cell is the basic unit of life. DNA's structure was discovered in 1953. Smoking increases cancer risk. Prevalence is 1 in 1000 births."

        Example Output:
        [
          {"id": 1, "assertion_subtype": "Definitional/Taxonomic", "statement_text": "The cell is the basic unit of life."},
          {"id": 2, "assertion_subtype": "Historical/Attributive", "statement_text": "DNA's structure was discovered in 1953."},
          {"id": 3, "assertion_subtype": "Causal/Mechanistic", "statement_text": "Smoking increases cancer risk."},
          {"id": 4, "assertion_subtype": "Quantitative/Statistical", "statement_text": "Prevalence is 1 in 1000 births."}
        ]

        Now analyze this input:
        [INPUT TEXT]
        
    """).strip()
    return base_prompt.replace("[INPUT TEXT]", input_text.strip())

def get_verification_prompt(assertions: str) -> str:
    """
    Return a prompt to verify the assertions using external sources.
    """
    base_prompt = dedent("""
        Prompt Template
        # Role and Goal
        You are a specialized AI assistant functioning as a rigorous scientific fact-checker. Your sole purpose is to verify a list of scientific assertions against specific, designated databases. Your operation must be completely transparent, and your responses must be based exclusively on the data you retrieve.

        # Core Directive
        For each scientific assertion provided in the list below, you will perform a search and analysis to determine if there is supporting evidence for it within the specified knowledge bases.

        # Knowledge Source Constraint (Absolute Rule)
        You are strictly sandboxed to the following two knowledge bases ONLY:

        NCBI Bookshelf

        PubMed

        You are absolutely forbidden from using any of your general pretrained knowledge or accessing any other internal or external information source. Your entire response for each assertion must be derived only from the contents of these two databases.

        # Step-by-Step Workflow

        Receive the list of scientific assertions.

        Process each assertion individually and sequentially.

        For a given assertion, formulate precise search queries to execute against the NCBI Bookshelf and PubMed databases.

        Analyze the search results to find direct, explicit evidence that supports or validates the assertion. The evidence must be a stated finding, conclusion, or data point in a published abstract, article, or book chapter.

        Based on the outcome of your analysis, generate a response for that assertion strictly following the output format defined below.

        # Output Formatting Requirements
        For each assertion, your output must contain the following components in order:

        Assertion: Repeat the original assertion verbatim.

        Status: Provide one of two possible statuses, formatted exactly as follows:

        [EVIDENCE FOUND]

        [NO EVIDENCE FOUND]

        Evidence/Conclusion:

        If the status is [EVIDENCE FOUND], provide a concise summary of the supporting evidence found. Whenever possible, include a direct quote from the source. You MUST include the citation(s), including the title, authors, journal/book, and the PubMed ID (PMID) or NCBI Bookshelf ID/link.

        If the status is [NO EVIDENCE FOUND], you must use the following exact phrase and nothing more: "A search of NCBI Bookshelf and PubMed did not yield direct supporting evidence for this assertion."

        # Critical Final Instruction
        If, for any technical reason, you cannot perform the search, or if the results are ambiguous and you are unsure if they constitute direct evidence, you MUST default to the [NO EVIDENCE FOUND] status and its corresponding response. Do not infer, guess, extrapolate, or apologize. Your only function is to report what is explicitly present in the designated sources. If the information is not there, you must state that you do not know by reporting [NO EVIDENCE FOUND].

        # Begin Fact-Checking Task: Assertions List

        [INPUT TEXT]
        
    """).strip()
    return base_prompt.replace("[INPUT TEXT]", assertions.strip())

def generate_content(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    """
    Generate content using the provided prompt and model.
    """
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text

def split_text_into_chunks(text: str, max_words: int = 2000) -> list:
    """
    Split text into chunks of approximately max_words each.
    Tries to split at paragraph boundaries to maintain context.
    """
    words = text.split()
    
    if len(words) <= max_words:
        return [text]
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        paragraph_word_count = len(paragraph_words)
        
        # If adding this paragraph would exceed the limit
        if current_word_count + paragraph_word_count > max_words and current_chunk:
            # Save current chunk and start a new one
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_word_count = paragraph_word_count
        else:
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_word_count += paragraph_word_count
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def extract_chapter_info(text: str) -> dict:
    """
    Extract chapter title and first few lines for identification.
    """
    lines = text.strip().split('\n')
    title = lines[0] if lines else "Unknown Chapter"
    
    # Get first few meaningful lines for preview
    preview_lines = []
    for line in lines[:10]:
        if line.strip() and not line.strip().startswith('#'):
            preview_lines.append(line.strip())
        if len(preview_lines) >= 3:
            break
    
    return {
        "title": title.replace('#', '').strip(),
        "preview": ' '.join(preview_lines)[:200] + "..." if preview_lines else ""
    }

def process_chapter_chunks(file_path: str, output_dir: str = None) -> dict:
    """
    Process a chapter file by splitting it into chunks and extracting assertions.
    Returns a summary of the processing results.
    """
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the chapter file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chapter_text = f.read()
    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}
    
    # Extract chapter information
    chapter_info = extract_chapter_info(chapter_text)
    chapter_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Split text into chunks
    chunks = split_text_into_chunks(chapter_text, max_words=2000)
    
    # Process each chunk
    results_summary = {
        "chapter_name": chapter_name,
        "chapter_title": chapter_info["title"],
        "total_chunks": len(chunks),
        "processed_chunks": [],
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nProcessing chunk {i}/{len(chunks)}...")
        
        # Generate chunk info
        chunk_info = {
            "chunk_number": i,
            "word_count": len(chunk.split()),
            "first_line": chunk.split('\n')[0][:100] + "..." if len(chunk.split('\n')[0]) > 100 else chunk.split('\n')[0],
            "last_line": chunk.split('\n')[-1][:100] + "..." if len(chunk.split('\n')[-1]) > 100 else chunk.split('\n')[-1]
        }
        
        try:
            # Extract assertions for this chunk
            assertion_prompt = get_assertion_prompt(chunk)
            assertions_response = generate_content(assertion_prompt)
            
            # Prepare output data
            output_data = {
                "metadata": {
                    "chapter_name": chapter_name,
                    "chapter_title": chapter_info["title"],
                    "chunk_number": i,
                    "total_chunks": len(chunks),
                    "chunk_info": chunk_info,
                    "processing_timestamp": datetime.now().isoformat(),
                    "word_count": len(chunk.split())
                },
                "input_text": chunk,
                "assertions_response": assertions_response
            }
            
            # Try to parse assertions as JSON for validation
            try:
                parsed_assertions = json.loads(assertions_response)
                output_data["parsed_assertions"] = parsed_assertions
                output_data["assertion_count"] = len(parsed_assertions) if isinstance(parsed_assertions, list) else 0
            except json.JSONDecodeError:
                output_data["parsing_error"] = "Failed to parse assertions as JSON"
                output_data["assertion_count"] = 0
            
            # Save to file
            output_filename = f"{chapter_name}_chunk_{i:02d}_assertions_{results_summary['timestamp']}.json"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Update results summary
            chunk_result = {
                "chunk_number": i,
                "output_file": output_filename,
                "word_count": len(chunk.split()),
                "assertion_count": output_data.get("assertion_count", 0),
                "status": "success",
                "preview": chunk_info
            }
            
            results_summary["processed_chunks"].append(chunk_result)
            print(f"✓ Chunk {i} processed successfully. Assertions: {output_data.get('assertion_count', 0)}")
            
        except Exception as e:
            error_msg = f"Error processing chunk {i}: {str(e)}"
            print(f"✗ {error_msg}")
            
            chunk_result = {
                "chunk_number": i,
                "word_count": len(chunk.split()),
                "status": "error",
                "error": error_msg,
                "preview": chunk_info
            }
            results_summary["processed_chunks"].append(chunk_result)
    
    # Save processing summary
    summary_filename = f"{chapter_name}_processing_summary_{results_summary['timestamp']}.json"
    summary_path = os.path.join(output_dir, summary_filename)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Processing complete!")
    print(f"📁 Total chunks processed: {len(chunks)}")
    print(f"📄 Summary saved to: {summary_filename}")
    print(f"📂 All files saved in: {output_dir}")
    
    return results_summary

def main():
    configure_sdk()
    
    # Configuration
    chapter_file_path = r"Popper _ Vibhor\Chapters\Chapter 02_ Introduction to Cancer_ A Disease of Deregulation.md"
    
    # Optional: specify output directory (if None, uses same directory as input file)
    output_directory = r"C:\Users\vibho\OneDrive\Desktop\Humanitarians.ai\Popper _ Vibhor\Chapter_02_Assertions"

    print("🚀 Starting automated assertion extraction...")
    print(f"📖 Processing file: {os.path.basename(chapter_file_path)}")
    print(f"📁 Output directory: {output_directory}")
    print("-" * 60)
    
    # Process the chapter
    results = process_chapter_chunks(chapter_file_path, output_directory)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Display summary
    print("\n" + "="*60)
    print("📊 PROCESSING SUMMARY")
    print("="*60)
    print(f"Chapter: {results['chapter_title']}")
    print(f"Total chunks: {results['total_chunks']}")
    
    successful_chunks = [c for c in results['processed_chunks'] if c['status'] == 'success']
    failed_chunks = [c for c in results['processed_chunks'] if c['status'] == 'error']
    
    print(f"✅ Successful: {len(successful_chunks)}")
    print(f"❌ Failed: {len(failed_chunks)}")
    
    if successful_chunks:
        total_assertions = sum(c.get('assertion_count', 0) for c in successful_chunks)
        print(f"📝 Total assertions extracted: {total_assertions}")
        
        print(f"\n📋 CHUNK BREAKDOWN:")
        for chunk in successful_chunks:
            print(f"  Chunk {chunk['chunk_number']:2d}: {chunk['assertion_count']:3d} assertions | {chunk['word_count']:4d} words | {chunk['output_file']}")
    
    if failed_chunks:
        print(f"\n⚠️  FAILED CHUNKS:")
        for chunk in failed_chunks:
            print(f"  Chunk {chunk['chunk_number']:2d}: {chunk['error']}")
    
    print(f"\n💾 All results saved in: {output_directory}")


if __name__ == '__main__':
    main()

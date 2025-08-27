#!/usr/bin/env python3
"""
Example script demonstrating how to use the Scientific Fact Checking System
"""

import json
import os
from pathlib import Path
from scientific_fact_checker import ScientificFactChecker, Config

def create_sample_chapter():
    """Create a sample chapter for demonstration"""
    sample_content = """
# Sample Chapter: Introduction to Cancer Biology

Cancer is a complex disease characterized by uncontrolled cell growth and proliferation. The fundamental abnormality resulting in the development of cancer is the continual unregulated proliferation of cancer cells.

## Key Concepts

### Cell Cycle Regulation
Normal cells follow a tightly regulated cell cycle that includes growth, DNA replication, and division phases. Cancer cells often bypass these regulatory checkpoints, leading to uncontrolled proliferation.

### Genetic Mutations
Cancer development is typically initiated by genetic mutations that affect key regulatory genes. These mutations can be inherited or acquired through environmental factors such as exposure to carcinogens.

### Tumor Formation
When cancer cells accumulate, they can form tumors. Benign tumors remain localized, while malignant tumors can invade surrounding tissues and metastasize to distant sites.

### Risk Factors
Several factors increase cancer risk, including tobacco use, excessive alcohol consumption, poor diet, lack of physical activity, and exposure to radiation or certain chemicals.

## Statistical Data
According to recent studies, approximately 40% of cancer cases are preventable through lifestyle modifications. The five-year survival rate for early-stage cancer detection is significantly higher than for late-stage detection.
"""
    
    # Create Chapters directory if it doesn't exist
    Path("Chapters").mkdir(exist_ok=True)
    
    # Write sample chapter
    with open("Chapters/sample_chapter.md", "w", encoding="utf-8") as f:
        f.write(sample_content)
    
    print("Sample chapter created: Chapters/sample_chapter.md")
    return "Chapters/sample_chapter.md"

def run_example():
    """Run the scientific fact checking system with sample data"""
    
    # Check if environment variables are set
    if not os.getenv('GOOGLE_API_KEY') or not os.getenv('TAVILY_API_KEY'):
        print("❌ Error: Please set your API keys in the .env file")
        print("1. Copy env_example.txt to .env")
        print("2. Add your Google Gemini and Tavily API keys")
        print("3. Run this script again")
        return
    
    try:
        # Create sample chapter
        chapter_path = create_sample_chapter()
        
        # Load configuration
        config = Config(
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            tavily_api_key=os.getenv('TAVILY_API_KEY'),
            search_domains=['ncbi.nlm.nih.gov', 'pubmed.ncbi.nlm.nih.gov'],
            batch_size=5,  # Smaller batch size for example
            chunk_size=500
        )
        
        # Initialize fact checker
        print("🔧 Initializing Scientific Fact Checker...")
        fact_checker = ScientificFactChecker(config)
        
        # Step 1: Extract assertions
        print("\n📝 Step 1: Extracting assertions from sample chapter...")
        assertions = fact_checker.extract_assertions(chapter_path)
        
        if not assertions:
            print("❌ No assertions extracted. Exiting.")
            return
        
        print(f"✅ Extracted {len(assertions)} testable assertions")
        
        # Show sample assertions
        print("\n📋 Sample assertions:")
        for i, assertion in enumerate(assertions[:3], 1):
            print(f"{i}. {assertion['Original Statement'][:100]}...")
        
        # Step 2: Find relevant documents (limited for example)
        print(f"\n🔍 Step 2: Finding relevant documents for {min(3, len(assertions))} assertions...")
        sample_assertions = assertions[:3]  # Limit to 3 for example
        documents_data = fact_checker.find_relevant_documents(sample_assertions)
        
        print(f"✅ Found relevant documents for {len(documents_data)} assertions")
        
        # Step 3: Build knowledge base
        print("\n📚 Step 3: Building knowledge base...")
        knowledge_base = fact_checker.build_knowledge_base(documents_data)
        
        if knowledge_base:
            print(f"✅ Knowledge base built with {len(knowledge_base)} characters")
        else:
            print("⚠️  Warning: Knowledge base is empty")
        
        # Step 4: Fact check statements
        print("\n🔬 Step 4: Fact checking statements...")
        results = fact_checker.fact_check_statements(documents_data, knowledge_base)
        
        print(f"✅ Fact checking completed for {len(results)} statements")
        
        # Step 5: Save results
        print("\n💾 Step 5: Saving results...")
        csv_path, json_path = fact_checker.save_results(results, "example_output")
        
        # Display summary
        print("\n📊 Results Summary:")
        verdict_counts = {}
        for result in results:
            verdict = result['Final Verdict']
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        for verdict, count in verdict_counts.items():
            print(f"  {verdict}: {count}")
        
        print(f"\n✅ Example completed successfully!")
        print(f"📁 Results saved to: {csv_path} and {json_path}")
        print(f"📋 Sample results:")
        
        # Show first result
        if results:
            first_result = results[0]
            print(f"\nStatement: {first_result['Statement'][:100]}...")
            print(f"Verdict: {first_result['Final Verdict']}")
        
    except Exception as e:
        print(f"❌ Error running example: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Scientific Fact Checking System - Example")
    print("=" * 50)
    run_example()

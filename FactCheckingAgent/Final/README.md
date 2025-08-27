# Scientific Fact Checking System

A comprehensive, AI-powered tool for extracting, validating, and fact-checking scientific statements using Google Gemini AI and Tavily search integration. This system is designed for academic and research applications, particularly in cancer biology and molecular biology.

## 🚀 Features

- **Automated Assertion Extraction**: Uses Gemini 2.5 Flash to extract testable scientific statements from markdown chapters
- **Intelligent Document Search**: Leverages Tavily API to find relevant peer-reviewed sources from NCBI and PubMed
- **RAG-Enhanced Fact Checking**: Combines Retrieval-Augmented Generation with Gemini AI for accurate fact verification
- **Batch Processing**: Efficiently processes multiple statements in batches to optimize API usage
- **Comprehensive Output**: Generates detailed analysis with clear verdicts (Correct, Incorrect, Flagged for Review)
- **Multiple Output Formats**: Saves results in both CSV and JSON formats for easy analysis
- **Configurable Parameters**: Customizable batch sizes, chunk sizes, and search domains
- **Robust Error Handling**: Comprehensive logging and graceful error recovery

## 📋 Prerequisites

- Python 3.8 or higher
- Google API key for Gemini (get from [Google AI Studio](https://makersuite.google.com/app/apikey))
- Tavily API key (get from [Tavily](https://tavily.com/))

## 🛠️ Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd scientific-fact-checker
   ```

2. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp env_example.txt .env
   
   # Edit .env with your API keys
   nano .env
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Required API Keys
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: Customize search domains (comma-separated)
SEARCH_DOMAINS=ncbi.nlm.nih.gov,pubmed.ncbi.nlm.nih.gov

# Optional: Customize batch size for processing
BATCH_SIZE=10

# Optional: Customize chunk size for RAG system
CHUNK_SIZE=500
```

### Getting API Keys

1. **Google Gemini API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the API key to your `.env` file

2. **Tavily API Key**:
   - Visit [Tavily](https://tavily.com/)
   - Sign up for an account
   - Get your API key from the dashboard
   - Copy the API key to your `.env` file

## 📁 Project Structure

```
scientific-fact-checker/
├── scientific_fact_checker.py    # Main application
├── requirements.txt              # Python dependencies
├── env_example.txt              # Environment variables template
├── README.md                    # This file
├── .env                         # Your API keys (create this)
├── Chapters/                    # Input markdown chapters
│   ├── Chapter 01_ The Building Blocks of Life_ Normal Cell Biology.md
│   ├── Chapter 02_ Introduction to Cancer_ A Disease of Deregulation.md
│   └── Chapter 03_ Cancer Epidemiology and Risk Factors.md
├── output/                      # Generated output files
│   ├── fact_checking_results.csv
│   └── detailed_fact_checking_results.json
└── logs/                        # Application logs
    └── fact_checker.log
```

## 🚀 Usage

### Basic Usage

1. **Prepare your input data**:
   - Place your markdown chapters in the `Chapters/` directory
   - Update the chapter path in `scientific_fact_checker.py` if needed

2. **Run the fact checking pipeline**:
   ```bash
   python scientific_fact_checker.py
   ```

3. **Review results**:
   - Check `output/fact_checking_results.csv` for the main results
   - Check `output/detailed_fact_checking_results.json` for detailed analysis
   - Review `fact_checker.log` for processing information

### Advanced Usage

#### Customizing Input Files

To process different chapters, modify the `chapter_path` variable in the `main()` function:

```python
# In scientific_fact_checker.py
chapter_path = "Chapters/Your_Chapter_Name.md"
```

#### Adjusting Processing Parameters

Modify the environment variables in your `.env` file:

```env
# Process more statements per batch (faster but uses more tokens)
BATCH_SIZE=20

# Larger chunks for RAG (more context but slower processing)
CHUNK_SIZE=1000

# Add more search domains
SEARCH_DOMAINS=ncbi.nlm.nih.gov,pubmed.ncbi.nlm.nih.gov,scholar.google.com
```

## 📊 Output Format

### CSV Output (`output/fact_checking_results.csv`)

Contains the main results with columns:
- **Statement**: Original statement from the chapter
- **Assertion**: Optimized assertion for search
- **Summary (Tavily Answer)**: AI-generated summary from Tavily
- **Relevant Docs**: List of relevant documents found
- **Final Verdict**: Fact checking result (Correct/Incorrect/Flagged for Review)

### JSON Output (`output/detailed_fact_checking_results.json`)

Contains detailed analysis including:
- All CSV columns
- **Full Analysis**: Complete Gemini response with reasoning and evidence assessment

### Intermediate Files

The system also generates intermediate files for debugging and analysis:
- `assertions_list.json`: Extracted testable assertions
- `relevant_documents.json`: Search results and document content
- `knowledge_base.txt`: Cleaned and processed knowledge base
- `rag_index.pkl`: RAG system index for efficient retrieval

## 🔍 Fact Checking Criteria

The system evaluates statements using three verdicts:

1. **Correct**: The statement is factually accurate and supported by the evidence
2. **Incorrect**: The statement contains factual errors or contradicts the evidence
3. **Flagged for Review**: The statement requires additional verification or contains ambiguous/contradictory information

## 🧠 How It Works

### 1. Assertion Extraction
- Uses Gemini 2.5 Flash to analyze markdown chapters sentence-by-sentence
- Identifies testable scientific statements
- Creates optimized assertions and search queries

### 2. Document Retrieval
- Searches NCBI and PubMed using Tavily API
- Extracts content from relevant scientific papers
- Builds a comprehensive knowledge base

### 3. RAG-Enhanced Fact Checking
- Uses TF-IDF vectorization to chunk the knowledge base
- Retrieves relevant chunks for each statement
- Combines RAG with Gemini AI for accurate fact verification

### 4. Batch Processing
- Processes statements in configurable batches
- Optimizes API usage and reduces costs
- Provides real-time progress updates

## 🛡️ Error Handling

The system includes comprehensive error handling:

- **API Key Validation**: Checks for required API keys on startup
- **File Loading**: Graceful handling of missing or corrupted files
- **API Rate Limiting**: Built-in delays to respect API limits
- **Network Errors**: Retry logic for temporary network issues
- **JSON Parsing**: Robust parsing of AI-generated responses
- **Logging**: Detailed logging for debugging and monitoring

## 🔧 Customization

### Adding New Data Sources

To add new search domains, update the `SEARCH_DOMAINS` environment variable:

```env
SEARCH_DOMAINS=ncbi.nlm.nih.gov,pubmed.ncbi.nlm.nih.gov,scholar.google.com,arxiv.org
```

### Modifying Fact Checking Logic

To customize the fact checking criteria, edit the prompt templates in the `ScientificFactChecker` class:

```python
# In _create_batch_fact_checking_prompt method
EVALUATION_CRITERIA = """
- Correct: The statement is factually accurate and supported by the evidence
- Incorrect: The statement contains factual errors or contradicts the evidence
- Flagged for Review: The statement requires additional verification
"""
```

### Adjusting RAG Parameters

To modify the RAG system behavior:

```python
# In the Config class
chunk_size = 1000  # Larger chunks for more context
overlap = 100      # More overlap for better continuity
```

## 📈 Performance Optimization

### API Cost Optimization

- **Batch Processing**: Reduces API calls by processing multiple statements together
- **Caching**: RAG index is saved and reused across runs
- **Rate Limiting**: Built-in delays prevent API rate limit issues

### Processing Speed

- **Parallel Processing**: Can be extended to process batches in parallel
- **Index Reuse**: RAG index is persisted between runs
- **Efficient Chunking**: Optimized text chunking for better retrieval

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**:
   ```
   ValueError: GOOGLE_API_KEY environment variable is required
   ```
   - Ensure your `.env` file exists and contains valid API keys
   - Check that the API keys have the necessary permissions

2. **File Not Found**:
   ```
   FileNotFoundError: [Errno 2] No such file or directory
   ```
   - Verify that the `Chapters/` directory exists
   - Check that the chapter file path is correct

3. **Rate Limiting**:
   ```
   Rate limit exceeded
   ```
   - The system includes built-in rate limiting
   - Increase delays in the code if needed

4. **Memory Issues**:
   - Reduce `BATCH_SIZE` for large datasets
   - Reduce `CHUNK_SIZE` for memory-constrained environments

### Getting Help

- Check the `fact_checker.log` file for detailed error information
- Review the console output for real-time processing status
- Ensure all dependencies are properly installed with `pip install -r requirements.txt`

## 📝 Logging

The system provides comprehensive logging:

- **File Logging**: All logs are saved to `fact_checker.log`
- **Console Output**: Real-time progress updates
- **Error Tracking**: Detailed error messages with stack traces
- **Performance Metrics**: Processing time and API usage statistics

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

For development, install additional dependencies:

```bash
pip install -r requirements.txt
pip install pytest black flake8 mypy
```

## 📄 License

This project is provided as-is for educational and research purposes. Please ensure compliance with the terms of service for Google Gemini and Tavily APIs.

## 🙏 Acknowledgments

- Google Gemini AI for providing the language model capabilities
- Tavily for web search and content extraction
- The scientific community for peer-reviewed research content
- Contributors and users who provide feedback and improvements

## 📞 Support

For questions, issues, or feature requests:

1. Check the troubleshooting section above
2. Review the logs for error details
3. Open an issue on the repository
4. Contact the development team

---

**Note**: This system is designed for academic and research use. Always verify results independently and consider the limitations of AI-based fact checking systems.

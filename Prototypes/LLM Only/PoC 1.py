import os
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

def main():
    configure_sdk()

    input_text = """
    Chapter 14: The Tumor Microenvironment
    14.1 Components of the Tumor Microenvironment
    The tumor microenvironment (TME) is a complex, dynamic ecosystem surrounding malignant cells that plays a crucial role in cancer initiation, progression, and metastasis. The TME, the environment surrounding the cancer cells, is a heterogeneous mixture of immune cells, endothelial cells, materials secreted from cells and their organelles, and fibroblasts. The tumor microenvironment (TME) is a complex biological structure surrounding tumor cells and includes blood vessels, immune cells, fibroblasts, adipocytes, and extracellular matrix (ECM).

    14.1.1 Cancer-Associated Fibroblasts
    Cancer-associated fibroblasts (CAFs) represent one of the most abundant and functionally important components of the tumor stroma. Cancer-associated fibroblasts (CAFs), a major component of the tumor microenvironment (TME), play an important role in cancer initiation, progression, and metastasis. Unlike normal fibroblasts, CAFs exhibit markedly different characteristics that promote tumorigenesis.

    Morphological and Functional Characteristics of CAFs

    CAFs, unlike normal fibroblasts (NF), are not passive bystanders. They possess similar characteristics to myofibroblasts, the fibroblasts responsible for wound healing and chronic inflammation, such as the expression of α-smooth muscle actin (α-SMA). Fibroblasts stem from a mesenchymal origin and have an elongated spindle or stellate shape with a multitude of cytoplasmic projections. Within the cytoplasm is an abundance of rough endoplasmic reticulum (rER) and a large Golgi apparatus.

    Origins and Activation of CAFs

    CAFs can originate from the activation and differentiation of quiescent fibroblasts, bone marrow-derived mesenchymal stem cells, and epithelial and endothelial cells. The transformation from normal fibroblasts to CAFs involves complex molecular programs that reprogram their phenotype and function.
    """

    assertion_prompt = get_assertion_prompt(input_text)
    assertions = generate_content(assertion_prompt)
    print("Assertions Response:\n", assertions)
    with open("assertions.txt", "w", encoding="utf-8") as f:
        f.write(assertions)
    
    verification_prompt = get_verification_prompt(assertions)
    verification = generate_content(verification_prompt)
    print("Verification Response:\n", verification)
    with open("verification.txt", "w", encoding="utf-8") as f:
        f.write(verification)
    

if __name__ == '__main__':
    main()

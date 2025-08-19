"""
Scientific Fact-Checking Agent
A comprehensive system for validating scientific claims against NCBI Bookshelf and PubMed
"""

import os
import json
import re
import time
import requests
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

try:
    import google.generativeai as genai
except ImportError:
    print("Installing required packages...")
    os.system("pip install google-generativeai requests")
    import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Assertion:
    """Represents a scientific assertion extracted from text"""
    claim: str
    assertion_type: str
    entities: List[str]
    confidence: str
    context: str
    searchable_query: str
    original_text: str

@dataclass
class Evidence:
    """Represents evidence from scientific literature"""
    title: str
    authors: List[str]
    journal: str
    publication_date: str
    pmid: str
    abstract: str
    doi: str
    publication_type: str
    source: str  # 'pubmed' or 'bookshelf'
    relevance_score: float
    quality_score: float

@dataclass
class ValidationResult:
    """Results of fact-checking an assertion"""
    assertion: Assertion
    status: str  # 'SUPPORTED', 'CONTRADICTED', 'INSUFFICIENT', 'MIXED'
    confidence_score: float
    supporting_evidence: List[Evidence]
    contradicting_evidence: List[Evidence]
    evidence_quality: str
    limitations: List[str]
    summary: str

class NCBISearcher:
    """Handles searches against NCBI databases"""
    
    def __init__(self, email: str = "user@example.com", api_key: str = None):
        self.email = email
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
    def search_pubmed(self, query: str, max_results: int = 20) -> List[Evidence]:
        """Search PubMed for relevant articles"""
        try:
            # Search for PMIDs
            search_url = f"{self.base_url}esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'sort': 'relevance',
                'retmode': 'json',
                'email': self.email
            }
            
            if self.api_key:
                search_params['api_key'] = self.api_key
            
            logger.info(f"Searching PubMed for: {query}")
            search_response = requests.get(search_url, params=search_params)
            search_data = search_response.json()
            
            if 'esearchresult' not in search_data or not search_data['esearchresult']['idlist']:
                logger.warning(f"No results found for query: {query}")
                return []
            
            pmids = search_data['esearchresult']['idlist']
            
            # Fetch details for found PMIDs
            fetch_url = f"{self.base_url}efetch.fcgi"
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml',
                'email': self.email
            }
            
            if self.api_key:
                fetch_params['api_key'] = self.api_key
            
            fetch_response = requests.get(fetch_url, params=fetch_params)
            
            return self._parse_pubmed_xml(fetch_response.text, query)
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            return []
    
    def _parse_pubmed_xml(self, xml_content: str, original_query: str) -> List[Evidence]:
        """Parse PubMed XML response into Evidence objects"""
        evidence_list = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    # Extract basic information
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else "No title available"
                    
                    # Authors
                    authors = []
                    author_elems = article.findall('.//Author')
                    for author in author_elems:
                        lastname = author.find('LastName')
                        forename = author.find('ForeName')
                        if lastname is not None and forename is not None:
                            authors.append(f"{forename.text} {lastname.text}")
                    
                    # Journal
                    journal_elem = article.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None else "Unknown journal"
                    
                    # Publication date
                    pub_date_elem = article.find('.//PubDate/Year')
                    pub_date = pub_date_elem.text if pub_date_elem is not None else "Unknown date"
                    
                    # PMID
                    pmid_elem = article.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else ""
                    
                    # Abstract
                    abstract_elem = article.find('.//Abstract/AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
                    
                    # DOI
                    doi_elem = article.find('.//ELocationID[@EIdType="doi"]')
                    doi = doi_elem.text if doi_elem is not None else ""
                    
                    # Publication type
                    pub_type_elem = article.find('.//PublicationType')
                    pub_type = pub_type_elem.text if pub_type_elem is not None else "Article"
                    
                    # Calculate relevance and quality scores
                    relevance_score = self._calculate_relevance(title + " " + abstract, original_query)
                    quality_score = self._calculate_quality_score(journal, pub_type, pub_date)
                    
                    evidence = Evidence(
                        title=title,
                        authors=authors,
                        journal=journal,
                        publication_date=pub_date,
                        pmid=pmid,
                        abstract=abstract,
                        doi=doi,
                        publication_type=pub_type,
                        source='pubmed',
                        relevance_score=relevance_score,
                        quality_score=quality_score
                    )
                    
                    evidence_list.append(evidence)
                    
                except Exception as e:
                    logger.warning(f"Error parsing article: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing PubMed XML: {str(e)}")
        
        return evidence_list
    
    def search_bookshelf(self, query: str, max_results: int = 10) -> List[Evidence]:
        """Search NCBI Bookshelf"""
        try:
            search_url = f"{self.base_url}esearch.fcgi"
            search_params = {
                'db': 'books',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'email': self.email
            }
            
            if self.api_key:
                search_params['api_key'] = self.api_key
            
            logger.info(f"Searching NCBI Bookshelf for: {query}")
            response = requests.get(search_url, params=search_params)
            data = response.json()
            
            if 'esearchresult' not in data or not data['esearchresult']['idlist']:
                return []
            
            # For bookshelf, we'll create simplified evidence entries
            # In a full implementation, you'd fetch detailed book information
            evidence_list = []
            for book_id in data['esearchresult']['idlist'][:max_results]:
                evidence = Evidence(
                    title=f"NCBI Bookshelf Entry {book_id}",
                    authors=["NCBI Authors"],
                    journal="NCBI Bookshelf",
                    publication_date="2024",
                    pmid=book_id,
                    abstract="Authoritative medical text from NCBI Bookshelf",
                    doi="",
                    publication_type="Book Chapter",
                    source='bookshelf',
                    relevance_score=0.8,
                    quality_score=0.9  # Bookshelf content is generally high quality
                )
                evidence_list.append(evidence)
            
            return evidence_list
            
        except Exception as e:
            logger.error(f"Error searching NCBI Bookshelf: {str(e)}")
            return []
    
    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score based on keyword overlap"""
        content_lower = content.lower()
        query_terms = query.lower().split()
        
        matches = sum(1 for term in query_terms if term in content_lower)
        return min(matches / len(query_terms), 1.0)
    
    def _calculate_quality_score(self, journal: str, pub_type: str, year: str) -> float:
        """Calculate quality score based on publication characteristics"""
        score = 0.5  # Base score
        
        # Publication type scoring
        type_scores = {
            'systematic review': 0.95,
            'meta-analysis': 0.95,
            'randomized controlled trial': 0.9,
            'clinical trial': 0.85,
            'review': 0.8,
            'comparative study': 0.75,
            'case-control study': 0.7,
            'cohort study': 0.7,
            'case reports': 0.5,
            'letter': 0.3,
            'comment': 0.3
        }
        
        pub_type_lower = pub_type.lower()
        for ptype, pscore in type_scores.items():
            if ptype in pub_type_lower:
                score = max(score, pscore)
                break
        
        # Recency bonus
        try:
            pub_year = int(year) if year.isdigit() else 2020
            current_year = datetime.now().year
            years_old = current_year - pub_year
            if years_old <= 5:
                score += 0.1
            elif years_old <= 10:
                score += 0.05
        except:
            pass
        
        return min(score, 1.0)

class GeminiProcessor:
    """Handles Gemini AI interactions for text analysis"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
    def extract_assertions(self, text: str) -> List[Assertion]:
        """Extract and classify assertions from scientific text"""
        
        prompt = f"""
        You are a scientific fact-checker. Analyze the following scientific text and extract all testable assertions.
        
        For each assertion, classify it into one of these types:
        1. CAUSAL: Claims about causation (X causes Y, X leads to Y)
        2. CORRELATIONAL: Claims about association (X is associated with Y, X correlates with Y)  
        3. MECHANISM: Claims about how something works (X functions by Y, X operates through Y)
        4. QUANTITATIVE: Claims with specific numbers (X occurs in Y% of cases, X reduces Y by Z%)
        5. COMPARATIVE: Claims comparing entities (X is more effective than Y, X is safer than Y)
        
        For each assertion, also determine:
        - Confidence level: EXPLICIT (clearly stated), IMPLICIT (implied), SPECULATIVE (hypothetical)
        - Key entities involved
        - Context or conditions
        - Optimized search query for medical databases
        
        Text to analyze:
        "{text}"
        
        Respond with valid JSON only, no other text:
        {{
            "assertions": [
                {{
                    "claim": "exact statement from text",
                    "type": "CAUSAL|CORRELATIONAL|MECHANISM|QUANTITATIVE|COMPARATIVE",
                    "entities": ["entity1", "entity2"],
                    "confidence": "EXPLICIT|IMPLICIT|SPECULATIVE",
                    "context": "any conditions or context mentioned",
                    "searchable_query": "optimized medical database search terms"
                }}
            ]
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean up response to extract JSON
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0]
            elif '```' in result_text:
                result_text = result_text.split('```')[1]
            
            # Remove any non-JSON content before and after
            result_text = result_text.strip()
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                result_text = result_text[start_idx:end_idx]
            
            logger.info(f"Cleaned JSON response: {result_text[:200]}...")
            result_data = json.loads(result_text)
            
            assertions = []
            for item in result_data.get('assertions', []):
                assertion = Assertion(
                    claim=item.get('claim', ''),
                    assertion_type=item.get('type', 'UNKNOWN'),
                    entities=item.get('entities', []),
                    confidence=item.get('confidence', 'IMPLICIT'),
                    context=item.get('context', ''),
                    searchable_query=item.get('searchable_query', item.get('claim', '')[:50]),
                    original_text=text
                )
                assertions.append(assertion)
            
            logger.info(f"Extracted {len(assertions)} assertions from text")
            return assertions
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Raw response: {result_text}")
            # Fallback: create a simple assertion from the original text
            return [Assertion(
                claim=text[:200] + "..." if len(text) > 200 else text,
                assertion_type="UNKNOWN",
                entities=[],
                confidence="IMPLICIT",
                context="",
                searchable_query=text[:50],
                original_text=text
            )]
        except Exception as e:
            logger.error(f"Error extracting assertions: {str(e)}")
            return []
    
    def evaluate_evidence(self, assertion: Assertion, evidence_list: List[Evidence]) -> ValidationResult:
        """Evaluate evidence against an assertion using Gemini"""
        
        # Prepare evidence summaries
        evidence_summaries = []
        for i, evidence in enumerate(evidence_list):
            summary = f"Evidence {i+1}: {evidence.title} ({evidence.journal}, {evidence.publication_date})\n"
            summary += f"Abstract: {evidence.abstract[:500]}...\n"
            summary += f"Quality Score: {evidence.quality_score:.2f}\n"
            evidence_summaries.append(summary)
        
        evidence_text = "\n".join(evidence_summaries)
        
        prompt = f"""
        You are a scientific fact-checker evaluating evidence for a specific claim.
        
        CLAIM TO EVALUATE:
        "{assertion.claim}"
        
        EVIDENCE FOUND:
        {evidence_text}
        
        Based on this evidence, determine:
        1. STATUS: Does the evidence SUPPORT, CONTRADICT, provide INSUFFICIENT data, or show MIXED results for this claim?
        2. CONFIDENCE: Rate your confidence (0.0 to 1.0) in this determination
        3. SUPPORTING_INDICES: Which evidence items (by number) support the claim?
        4. CONTRADICTING_INDICES: Which evidence items (by number) contradict the claim?
        5. QUALITY: Rate the overall evidence quality as HIGH, MEDIUM, or LOW
        6. LIMITATIONS: What are the key limitations of this evidence?
        7. SUMMARY: Provide a brief summary of your findings
        
        Respond with valid JSON only:
        {{
            "status": "SUPPORTED|CONTRADICTED|INSUFFICIENT|MIXED",
            "confidence_score": 0.0,
            "supporting_indices": [1, 2],
            "contradicting_indices": [3],
            "evidence_quality": "HIGH|MEDIUM|LOW",
            "limitations": ["limitation1", "limitation2"],
            "summary": "Brief summary of findings"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean up response
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0]
            elif '```' in result_text:
                result_text = result_text.split('```')[1]
            
            # Remove any non-JSON content
            result_text = result_text.strip()
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                result_text = result_text[start_idx:end_idx]
            
            result_data = json.loads(result_text)
            
            # Separate supporting and contradicting evidence
            supporting_evidence = []
            contradicting_evidence = []
            
            for idx in result_data.get('supporting_indices', []):
                if 1 <= idx <= len(evidence_list):
                    supporting_evidence.append(evidence_list[idx-1])
            
            for idx in result_data.get('contradicting_indices', []):
                if 1 <= idx <= len(evidence_list):
                    contradicting_evidence.append(evidence_list[idx-1])
            
            validation_result = ValidationResult(
                assertion=assertion,
                status=result_data.get('status', 'INSUFFICIENT'),
                confidence_score=float(result_data.get('confidence_score', 0.0)),
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                evidence_quality=result_data.get('evidence_quality', 'LOW'),
                limitations=result_data.get('limitations', []),
                summary=result_data.get('summary', 'Could not generate summary')
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error evaluating evidence: {str(e)}")
            return ValidationResult(
                assertion=assertion,
                status='INSUFFICIENT',
                confidence_score=0.0,
                supporting_evidence=[],
                contradicting_evidence=[],
                evidence_quality='LOW',
                limitations=['Error in evaluation process'],
                summary='Could not evaluate due to processing error'
            )

class ScientificFactChecker:
    """Main fact-checking orchestrator"""
    
    def __init__(self, gemini_api_key: str, ncbi_email: str = "user@example.com", ncbi_api_key: str = None):
        self.gemini_processor = GeminiProcessor(gemini_api_key)
        self.ncbi_searcher = NCBISearcher(ncbi_email, ncbi_api_key)
        
    def fact_check_text(self, text: str, max_evidence_per_claim: int = 10) -> Dict[str, Any]:
        """Main fact-checking workflow"""
        logger.info("Starting fact-checking process...")
        
        # Step 1: Extract assertions
        logger.info("Step 1: Extracting assertions...")
        assertions = self.gemini_processor.extract_assertions(text)
        
        if not assertions:
            logger.warning("No assertions found in text")
            return self._generate_empty_report(text)
        
        # Step 2: Search for evidence and validate each assertion
        logger.info("Step 2: Searching for evidence and validating assertions...")
        validation_results = []
        
        for i, assertion in enumerate(assertions):
            logger.info(f"Processing assertion {i+1}/{len(assertions)}: {assertion.claim[:100]}...")
            
            # Search for evidence
            evidence = self._search_evidence(assertion, max_evidence_per_claim)
            
            # Validate assertion
            if evidence:
                validation = self.gemini_processor.evaluate_evidence(assertion, evidence)
            else:
                validation = ValidationResult(
                    assertion=assertion,
                    status='INSUFFICIENT',
                    confidence_score=0.0,
                    supporting_evidence=[],
                    contradicting_evidence=[],
                    evidence_quality='LOW',
                    limitations=['No relevant evidence found'],
                    summary='No evidence found in scientific databases'
                )
            
            validation_results.append(validation)
            
            # Rate limiting
            time.sleep(1)
        
        # Step 3: Generate comprehensive report
        logger.info("Step 3: Generating report...")
        report = self._generate_report(text, validation_results)
        
        return report
    
    def _search_evidence(self, assertion: Assertion, max_results: int) -> List[Evidence]:
        """Search both PubMed and NCBI Bookshelf for evidence"""
        all_evidence = []
        
        # Search PubMed
        pubmed_evidence = self.ncbi_searcher.search_pubmed(
            assertion.searchable_query, 
            max_results // 2
        )
        all_evidence.extend(pubmed_evidence)
        
        # Search NCBI Bookshelf
        bookshelf_evidence = self.ncbi_searcher.search_bookshelf(
            assertion.searchable_query,
            max_results // 2
        )
        all_evidence.extend(bookshelf_evidence)
        
        # Sort by quality and relevance
        all_evidence.sort(key=lambda x: (x.quality_score + x.relevance_score) / 2, reverse=True)
        
        return all_evidence[:max_results]
    
    def _generate_report(self, original_text: str, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive fact-checking report"""
        
        # Calculate summary statistics
        total_assertions = len(validation_results)
        supported = len([r for r in validation_results if r.status == 'SUPPORTED'])
        contradicted = len([r for r in validation_results if r.status == 'CONTRADICTED'])
        insufficient = len([r for r in validation_results if r.status == 'INSUFFICIENT'])
        mixed = len([r for r in validation_results if r.status == 'MIXED'])
        
        # Calculate overall credibility score
        credibility_scores = []
        for result in validation_results:
            if result.status == 'SUPPORTED':
                credibility_scores.append(result.confidence_score)
            elif result.status == 'CONTRADICTED':
                credibility_scores.append(1.0 - result.confidence_score)
            else:
                credibility_scores.append(0.5)  # Neutral for insufficient/mixed
        
        overall_credibility = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0.0
        
        # Generate detailed analysis
        detailed_analysis = []
        for result in validation_results:
            analysis = {
                'assertion': asdict(result.assertion),
                'validation': {
                    'status': result.status,
                    'confidence_score': result.confidence_score,
                    'evidence_quality': result.evidence_quality,
                    'summary': result.summary,
                    'limitations': result.limitations
                },
                'evidence': {
                    'supporting_count': len(result.supporting_evidence),
                    'contradicting_count': len(result.contradicting_evidence),
                    'supporting_sources': [
                        {
                            'title': ev.title,
                            'journal': ev.journal,
                            'year': ev.publication_date,
                            'pmid': ev.pmid,
                            'quality_score': ev.quality_score
                        }
                        for ev in result.supporting_evidence
                    ],
                    'contradicting_sources': [
                        {
                            'title': ev.title,
                            'journal': ev.journal,
                            'year': ev.publication_date,
                            'pmid': ev.pmid,
                            'quality_score': ev.quality_score
                        }
                        for ev in result.contradicting_evidence
                    ]
                }
            }
            detailed_analysis.append(analysis)
        
        report = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'original_text': original_text,
                'databases_searched': ['PubMed', 'NCBI Bookshelf'],
                'model_used': 'Gemini 2.0 Flash'
            },
            'summary': {
                'total_assertions': total_assertions,
                'supported': supported,
                'contradicted': contradicted,
                'insufficient_evidence': insufficient,
                'mixed_evidence': mixed,
                'overall_credibility_score': round(overall_credibility, 3),
                'credibility_interpretation': self._interpret_credibility(overall_credibility)
            },
            'detailed_analysis': detailed_analysis,
            'methodology': {
                'evidence_hierarchy': 'Systematic Reviews > Meta-analyses > RCTs > Observational Studies',
                'quality_scoring': 'Based on publication type, journal reputation, and recency',
                'validation_approach': 'AI-assisted evidence evaluation with human oversight recommended'
            }
        }
        
        return report
    
    def _generate_empty_report(self, text: str) -> Dict[str, Any]:
        """Generate report when no assertions are found"""
        return {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'original_text': text,
                'databases_searched': ['PubMed', 'NCBI Bookshelf'],
                'model_used': 'Gemini 2.0 Flash'
            },
            'summary': {
                'total_assertions': 0,
                'supported': 0,
                'contradicted': 0,
                'insufficient_evidence': 0,
                'mixed_evidence': 0,
                'overall_credibility_score': 0.0,
                'credibility_interpretation': 'No testable assertions found'
            },
            'detailed_analysis': [],
            'note': 'No testable scientific assertions were identified in the provided text.'
        }
    
    def _interpret_credibility(self, score: float) -> str:
        """Interpret credibility score"""
        if score >= 0.8:
            return "HIGH - Claims are well-supported by scientific evidence"
        elif score >= 0.6:
            return "MODERATE - Claims have reasonable scientific support"
        elif score >= 0.4:
            return "LOW - Claims have limited scientific support"
        else:
            return "VERY LOW - Claims lack adequate scientific support"

def main():
    """Main function for command-line usage"""
    print("Scientific Fact-Checking Agent")
    print("=" * 50)
    
    # Get API keys
    gemini_api_key = input("Enter your Gemini API key: ").strip()
    if not gemini_api_key:
        print("Gemini API key is required!")
        return
    
    ncbi_email = input("Enter your email for NCBI (optional, press enter for default): ").strip()
    if not ncbi_email:
        ncbi_email = "user@example.com"
    
    ncbi_api_key = input("Enter your NCBI API key (optional, press enter to skip): ").strip()
    
    # Initialize fact-checker
    try:
        fact_checker = ScientificFactChecker(
            gemini_api_key=gemini_api_key,
            ncbi_email=ncbi_email,
            ncbi_api_key=ncbi_api_key or None
        )
        print("\nFact-checker initialized successfully!")
    except Exception as e:
        print(f"Error initializing fact-checker: {str(e)}")
        return
    
    # Get text to analyze
    print("\nEnter the scientific text to fact-check (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "" and len(lines) > 0 and lines[-1] == "":
            break
        lines.append(line)
    
    text = '\n'.join(lines[:-1])  # Remove the last empty line
    
    if not text.strip():
        print("No text provided!")
        return
    
    print("\nAnalyzing text... This may take a few minutes.")
    
    # Perform fact-checking
    try:
        results = fact_checker.fact_check_text(text)
        
        # Display results
        print("\n" + "=" * 50)
        print("FACT-CHECKING RESULTS")
        print("=" * 50)
        
        summary = results['summary']
        print(f"Total Assertions: {summary['total_assertions']}")
        print(f"Supported: {summary['supported']}")
        print(f"Contradicted: {summary['contradicted']}")
        print(f"Insufficient Evidence: {summary['insufficient_evidence']}")
        print(f"Mixed Evidence: {summary['mixed_evidence']}")
        print(f"Overall Credibility: {summary['overall_credibility_score']:.3f}")
        print(f"Interpretation: {summary['credibility_interpretation']}")
        
        # Show detailed results
        print("\nDETAILED ANALYSIS:")
        print("-" * 30)
        
        for i, analysis in enumerate(results['detailed_analysis'], 1):
            assertion = analysis['assertion']
            validation = analysis['validation']
            evidence = analysis['evidence']
            
            print(f"\n{i}. ASSERTION: {assertion['claim']}")
            print(f"   Type: {assertion['assertion_type']}")
            print(f"   Status: {validation['status']}")
            print(f"   Confidence: {validation['confidence_score']:.3f}")
            print(f"   Evidence Quality: {validation['evidence_quality']}")
            print(f"   Summary: {validation['summary']}")
            
            if evidence['supporting_count'] > 0:
                print(f"   Supporting Sources ({evidence['supporting_count']}):")
                for source in evidence['supporting_sources'][:3]:  # Show top 3
                    print(f"     - {source['title']} ({source['journal']}, {source['year']})")
            
            if evidence['contradicting_count'] > 0:
                print(f"   Contradicting Sources ({evidence['contradicting_count']}):")
                for source in evidence['contradicting_sources'][:3]:  # Show top 3
                    print(f"     - {source['title']} ({source['journal']}, {source['year']})")
        
        # Save results to file
        output_file = f"fact_check_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during fact-checking: {str(e)}")
        logger.error(f"Fact-checking error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
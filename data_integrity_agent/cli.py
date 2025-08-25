#!/usr/bin/env python3
"""
Command-line interface for the Data Integrity Agent
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import List, Optional
import pandas as pd
from tqdm import tqdm
import colorama
from colorama import Fore, Style

# Import our agents
from data_integrity_agent import BasicDataIntegrityAgent
from llm_enhanced_agent import LLMEnhancedDataIntegrityAgent

# Initialize colorama for cross-platform colored output
colorama.init()

class DataIntegrityCLI:
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the command-line argument parser"""
        parser = argparse.ArgumentParser(
            description="Data Integrity Agent - Validate dataset quality and integrity",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic validation
  python cli.py validate data.csv
  
  # LLM-enhanced validation
  python cli.py validate data.csv --llm
  
  # Batch validation
  python cli.py validate-batch data/ --output results/
  
  # Generate report
  python cli.py validate data.csv --output report.html --format html
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Validate command
        validate_parser = subparsers.add_parser('validate', help='Validate a single dataset')
        validate_parser.add_argument('file', help='Path to the dataset file (CSV, Excel, etc.)')
        validate_parser.add_argument('--llm', action='store_true', help='Use LLM-enhanced validation')

        validate_parser.add_argument('--output', help='Output file path for results')
        validate_parser.add_argument('--format', choices=['json', 'html', 'text'], default='json', 
                                   help='Output format')
        validate_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        
        # Batch validation command
        batch_parser = subparsers.add_parser('validate-batch', help='Validate multiple datasets')
        batch_parser.add_argument('input_dir', help='Directory containing datasets')
        batch_parser.add_argument('--output', required=True, help='Output directory for results')
        batch_parser.add_argument('--llm', action='store_true', help='Use LLM-enhanced validation')

        batch_parser.add_argument('--pattern', default='*.csv', help='File pattern to match')
        
        # List supported formats
        list_parser = subparsers.add_parser('list-formats', help='List supported file formats')
        
        return parser
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI with given arguments"""
        parsed_args = self.parser.parse_args(args)
        
        if not parsed_args.command:
            self.parser.print_help()
            return 1
        
        try:
            if parsed_args.command == 'validate':
                return self._validate_single(parsed_args)
            elif parsed_args.command == 'validate-batch':
                return self._validate_batch(parsed_args)
            elif parsed_args.command == 'list-formats':
                return self._list_formats()
            else:
                print(f"{Fore.RED}Unknown command: {parsed_args.command}{Style.RESET_ALL}")
                return 1
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Operation cancelled by user{Style.RESET_ALL}")
            return 1
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            if parsed_args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _validate_single(self, args) -> int:
        """Validate a single dataset"""
        file_path = Path(args.file)
        
        if not file_path.exists():
            print(f"{Fore.RED}Error: File not found: {file_path}{Style.RESET_ALL}")
            return 1
        
        print(f"{Fore.CYAN}Loading dataset: {file_path}{Style.RESET_ALL}")
        
        try:
            df = self._load_dataset(file_path)
        except Exception as e:
            print(f"{Fore.RED}Error loading dataset: {e}{Style.RESET_ALL}")
            return 1
        
        print(f"{Fore.GREEN}Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns{Style.RESET_ALL}")
        
        # Initialize agent
        if args.llm:
            print(f"{Fore.CYAN}Initializing LLM-enhanced agent...{Style.RESET_ALL}")
            try:
                agent = LLMEnhancedDataIntegrityAgent()
                results = agent.intelligent_validate_dataset(df)
            except Exception as e:
                print(f"{Fore.RED}LLM agent failed: {e}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Falling back to basic agent...{Style.RESET_ALL}")
                agent = BasicDataIntegrityAgent()
                results = agent.validate_dataset(df)
        else:
            print(f"{Fore.CYAN}Running basic validation...{Style.RESET_ALL}")
            agent = BasicDataIntegrityAgent()
            results = agent.validate_dataset(df)
        
        # Display results
        self._display_results(results, args)
        
        # Save results if output specified
        if args.output:
            self._save_results(results, args.output, args.format)
        
        return 0
    
    def _validate_batch(self, args) -> int:
        """Validate multiple datasets in batch"""
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output)
        
        if not input_dir.exists():
            print(f"{Fore.RED}Error: Input directory not found: {input_dir}{Style.RESET_ALL}")
            return 1
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find files matching pattern
        files = list(input_dir.glob(args.pattern))
        if not files:
            print(f"{Fore.YELLOW}No files found matching pattern: {args.pattern}{Style.RESET_ALL}")
            return 1
        
        print(f"{Fore.CYAN}Found {len(files)} files to validate{Style.RESET_ALL}")
        
        # Initialize agent
        if args.llm:
            try:
                agent = LLMEnhancedDataIntegrityAgent()
                use_llm = True
            except Exception as e:
                print(f"{Fore.YELLOW}LLM agent not available, using basic agent: {e}{Style.RESET_ALL}")
                agent = BasicDataIntegrityAgent()
                use_llm = False
        else:
            agent = BasicDataIntegrityAgent()
            use_llm = False
        
        # Process files
        results_summary = []
        
        for file_path in tqdm(files, desc="Validating datasets", unit="file"):
            try:
                df = self._load_dataset(file_path)
                
                if use_llm:
                    results = agent.intelligent_validate_dataset(df)
                else:
                    results = agent.validate_dataset(df)
                
                # Add file info to results
                results['file_info'] = {
                    'filename': file_path.name,
                    'path': str(file_path),
                    'rows': len(df),
                    'columns': len(df.columns)
                }
                
                # Save individual results
                output_file = output_dir / f"{file_path.stem}_validation.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                results_summary.append(results)
                
            except Exception as e:
                print(f"{Fore.RED}Error processing {file_path.name}: {e}{Style.RESET_ALL}")
                continue
        
        # Generate summary report
        summary_file = output_dir / "batch_validation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"{Fore.GREEN}Batch validation complete! Results saved to: {output_dir}{Style.RESET_ALL}")
        return 0
    
    def _list_formats(self) -> int:
        """List supported file formats"""
        print(f"{Fore.CYAN}Supported file formats:{Style.RESET_ALL}")
        formats = [
            ("CSV", ".csv", "Comma-separated values"),
            ("Excel", ".xlsx, .xls", "Microsoft Excel files"),
            ("JSON", ".json", "JavaScript Object Notation"),
            ("Parquet", ".parquet", "Columnar storage format"),
            ("HDF5", ".h5, .hdf5", "Hierarchical Data Format"),
            ("Pickle", ".pkl, .pickle", "Python pickle format")
        ]
        
        for name, extensions, description in formats:
            print(f"  {Fore.GREEN}{name}{Style.RESET_ALL}: {extensions} - {description}")
        
        return 0
    
    def _load_dataset(self, file_path: Path) -> pd.DataFrame:
        """Load dataset from file"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            return pd.read_csv(file_path)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif suffix == '.json':
            return pd.read_json(file_path)
        elif suffix == '.parquet':
            return pd.read_parquet(file_path)
        elif suffix in ['.h5', '.hdf5']:
            return pd.read_hdf(file_path)
        elif suffix in ['.pkl', '.pickle']:
            return pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _display_results(self, results: dict, args) -> None:
        """Display validation results"""
        print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}VALIDATION RESULTS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        
        if 'data_quality_score' in results:
            score = results['data_quality_score']
            if score >= 90:
                color = Fore.GREEN
                status = "EXCELLENT"
            elif score >= 70:
                color = Fore.YELLOW
                status = "GOOD"
            elif score >= 50:
                color = Fore.RED
                status = "MODERATE"
            else:
                color = Fore.RED
                status = "POOR"
            
            print(f"{color}Data Quality Score: {score}/100 ({status}){Style.RESET_ALL}")
        
        if 'validation_results' in results:
            validation = results['validation_results']
            
            if 'missing_values' in validation:
                missing = validation['missing_values']
                print(f"\n{Fore.YELLOW}Missing Values:{Style.RESET_ALL}")
                print(f"  Total missing: {missing.get('total_missing', 0)}")
                print(f"  Percentage: {missing.get('total_percentage', 0):.2f}%")
            
            if 'outliers' in validation:
                outliers = validation['outliers']
                print(f"\n{Fore.YELLOW}Outliers:{Style.RESET_ALL}")
                for col, info in outliers.items():
                    print(f"  {col}: {info['count']} ({info['percentage']:.2f}%)")
        
        if 'recommendations' in results:
            print(f"\n{Fore.GREEN}Recommendations:{Style.RESET_ALL}")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        if args.verbose:
            print(f"\n{Fore.CYAN}Detailed Results:{Style.RESET_ALL}")
            print(json.dumps(results, indent=2))
    
    def _save_results(self, results: dict, output_path: str, format_type: str) -> None:
        """Save results to file"""
        output_path = Path(output_path)
        
        if format_type == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif format_type == 'html':
            self._save_html_report(results, output_path)
        elif format_type == 'text':
            with open(output_path, 'w') as f:
                f.write(self._format_text_report(results))
        
        print(f"{Fore.GREEN}Results saved to: {output_path}{Style.RESET_ALL}")
    
    def _save_html_report(self, results: dict, output_path: Path) -> None:
        """Save results as HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Integrity Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .score {{ font-size: 24px; font-weight: bold; }}
        .good {{ color: green; }}
        .moderate {{ color: orange; }}
        .poor {{ color: red; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Data Integrity Validation Report</h1>
        <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Summary</h2>
        <p class="score">Data Quality Score: {results.get('data_quality_score', 'N/A')}/100</p>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
"""
        
        for rec in results.get('recommendations', []):
            html_content += f"            <li>{rec}</li>\n"
        
        html_content += """
        </ul>
    </div>
    
    <div class="section">
        <h2>Detailed Results</h2>
        <pre>"""
        
        html_content += json.dumps(results, indent=2)
        
        html_content += """
        </pre>
    </div>
</body>
</html>"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _format_text_report(self, results: dict) -> str:
        """Format results as text report"""
        report = []
        report.append("=" * 50)
        report.append("DATA INTEGRITY VALIDATION REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if 'data_quality_score' in results:
            report.append(f"Data Quality Score: {results['data_quality_score']}/100")
            report.append("")
        
        if 'recommendations' in results:
            report.append("RECOMMENDATIONS:")
            for i, rec in enumerate(results['recommendations'], 1):
                report.append(f"  {i}. {rec}")
            report.append("")
        
        report.append("DETAILED RESULTS:")
        report.append(json.dumps(results, indent=2))
        
        return "\n".join(report)

def main():
    """Main entry point"""
    cli = DataIntegrityCLI()
    sys.exit(cli.run())

if __name__ == "__main__":
    main()

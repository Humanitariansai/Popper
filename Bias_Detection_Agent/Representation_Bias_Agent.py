"""
Representation Bias Agent

Purpose: Assess whether current dataset reflects diverse populations
Capabilities:
- Demographic distribution analysis
- Under/over-representation quantification  
- Inclusion gap identification
- Historical context evaluation
- Diversity metric calculation and tracking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class RepresentationBiasAgent:
    def __init__(self):
        """Initialize the Representation Bias Agent"""
        self.dataset = None
        self.protected_attributes = []
        self.population_benchmarks = {
            # US Census 2020 approximate demographics for comparison
            'gender': {
                'Male': 0.494,
                'Female': 0.506
            },
            'race': {
                'White': 0.722,
                'Black': 0.134,
                'Asian-Pac-Islander': 0.061,
                'Amer-Indian-Eskimo': 0.013,
                'Other': 0.070
            },
            'age_groups': {
                '18-24': 0.098,
                '25-34': 0.135,
                '35-44': 0.128,
                '45-54': 0.129,
                '55-64': 0.126,
                '65+': 0.166,
                'Under-18': 0.218  # Note: Adult dataset excludes this
            }
        }
    
    def load_dataset(self, data_path: str, protected_attrs: List[str] = None):
        """Load dataset and set protected attributes"""
        try:
            self.dataset = pd.read_csv(data_path)
            if protected_attrs:
                self.protected_attributes = protected_attrs
            else:
                # Auto-detect common protected attributes
                self.protected_attributes = [col for col in self.dataset.columns 
                                           if col.lower() in ['gender', 'sex', 'race', 'age', 'ethnicity']]
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def analyze_demographic_distribution(self) -> Dict[str, Any]:
        """Analyze demographic distribution of the dataset"""
        if self.dataset is None:
            return {"error": "No dataset loaded"}
        
        analysis = {}
        
        for attr in self.protected_attributes:
            if attr in self.dataset.columns:
                # Get value counts and proportions
                value_counts = self.dataset[attr].value_counts()
                proportions = self.dataset[attr].value_counts(normalize=True)
                
                analysis[attr] = {
                    'counts': value_counts.to_dict(),
                    'proportions': proportions.to_dict(),
                    'unique_values': len(value_counts),
                    'most_common': value_counts.index[0],
                    'least_common': value_counts.index[-1],
                    'dominance_ratio': value_counts.iloc[0] / value_counts.iloc[-1] if len(value_counts) > 1 else 1.0
                }
        
        return analysis
    
    def quantify_representation_gaps(self) -> Dict[str, Any]:
        """Quantify under/over-representation compared to population benchmarks"""
        if self.dataset is None:
            return {"error": "No dataset loaded"}
        
        gaps = {}
        
        for attr in self.protected_attributes:
            if attr in self.dataset.columns and attr in self.population_benchmarks:
                observed = self.dataset[attr].value_counts(normalize=True).to_dict()
                expected = self.population_benchmarks[attr]
                
                attr_gaps = {}
                for group, expected_prop in expected.items():
                    observed_prop = observed.get(group, 0.0)
                    representation_ratio = observed_prop / expected_prop if expected_prop > 0 else float('inf')
                    gap = observed_prop - expected_prop
                    
                    attr_gaps[group] = {
                        'observed_proportion': observed_prop,
                        'expected_proportion': expected_prop,
                        'representation_ratio': representation_ratio,
                        'absolute_gap': gap,
                        'relative_gap_percent': (gap / expected_prop * 100) if expected_prop > 0 else float('inf'),
                        'status': self._classify_representation_status(representation_ratio)
                    }
                
                gaps[attr] = attr_gaps
        
        return gaps
    
    def _classify_representation_status(self, ratio: float) -> str:
        """Classify representation status based on ratio"""
        if ratio < 0.7:
            return "Severely Under-represented"
        elif ratio < 0.85:
            return "Under-represented"
        elif ratio > 1.3:
            return "Over-represented"
        elif ratio > 1.15:
            return "Slightly Over-represented"
        else:
            return "Adequately Represented"
    
    def identify_inclusion_gaps(self) -> Dict[str, Any]:
        """Identify specific inclusion gaps and missing demographics"""
        if self.dataset is None:
            return {"error": "No dataset loaded"}
        
        inclusion_analysis = {}
        
        for attr in self.protected_attributes:
            if attr in self.dataset.columns:
                observed_groups = set(self.dataset[attr].unique())
                
                if attr in self.population_benchmarks:
                    expected_groups = set(self.population_benchmarks[attr].keys())
                    missing_groups = expected_groups - observed_groups
                    unexpected_groups = observed_groups - expected_groups
                    
                    inclusion_analysis[attr] = {
                        'observed_groups': list(observed_groups),
                        'expected_groups': list(expected_groups),
                        'missing_groups': list(missing_groups),
                        'unexpected_groups': list(unexpected_groups),
                        'coverage_ratio': len(observed_groups & expected_groups) / len(expected_groups),
                        'inclusion_score': self._calculate_inclusion_score(attr)
                    }
        
        return inclusion_analysis
    
    def _calculate_inclusion_score(self, attr: str) -> float:
        """Calculate inclusion score for an attribute (0-1 scale)"""
        if attr not in self.dataset.columns or attr not in self.population_benchmarks:
            return 0.0
        
        observed = self.dataset[attr].value_counts(normalize=True).to_dict()
        expected = self.population_benchmarks[attr]
        
        # Calculate weighted inclusion score based on representation accuracy
        total_score = 0.0
        total_weight = 0.0
        
        for group, expected_prop in expected.items():
            observed_prop = observed.get(group, 0.0)
            if expected_prop > 0:
                # Score based on how close observed is to expected
                accuracy = 1 - abs(observed_prop - expected_prop) / expected_prop
                weight = expected_prop  # Weight by expected proportion
                total_score += accuracy * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def calculate_diversity_metrics(self) -> Dict[str, Any]:
        """Calculate various diversity metrics"""
        if self.dataset is None:
            return {"error": "No dataset loaded"}
        
        metrics = {}
        
        for attr in self.protected_attributes:
            if attr in self.dataset.columns:
                proportions = self.dataset[attr].value_counts(normalize=True)
                
                # Shannon Diversity Index (H')
                shannon_diversity = -sum(p * np.log(p) for p in proportions if p > 0)
                
                # Simpson's Diversity Index (1-D)
                simpson_diversity = 1 - sum(p**2 for p in proportions)
                
                # Berger-Parker Dominance Index
                berger_parker = proportions.max()
                
                # Effective Number of Groups (Hill's N1)
                effective_groups = np.exp(shannon_diversity)
                
                # Evenness Index (Pielou's J')
                max_diversity = np.log(len(proportions))
                evenness = shannon_diversity / max_diversity if max_diversity > 0 else 0
                
                # Gini-Simpson Index (alternative formulation)
                gini_simpson = 1 - sum(p * (p - 1) for p in proportions) / (1 - 1/len(proportions)) if len(proportions) > 1 else 0
                
                # Inverse Simpson Index
                inverse_simpson = 1 / sum(p**2 for p in proportions) if sum(p**2 for p in proportions) > 0 else 0
                
                # Rényi Diversity (order 2)
                renyi_2 = np.log(inverse_simpson)
                
                # True Diversity (Hill numbers)
                hill_0 = len(proportions)  # Species richness
                hill_1 = effective_groups  # Shannon diversity exponential
                hill_2 = inverse_simpson   # Simpson diversity reciprocal
                
                # Normalized diversity metrics
                max_possible_shannon = np.log(len(proportions))
                shannon_evenness = shannon_diversity / max_possible_shannon if max_possible_shannon > 0 else 0
                
                # Inequality measures
                theil_index = sum(p * np.log(p * len(proportions)) for p in proportions if p > 0)
                
                # Diversity classification
                diversity_level = self._classify_diversity_level(shannon_diversity, len(proportions))
                
                metrics[attr] = {
                    # Core diversity indices
                    'shannon_diversity': shannon_diversity,
                    'simpson_diversity': simpson_diversity,
                    'gini_simpson': gini_simpson,
                    'inverse_simpson': inverse_simpson,
                    'berger_parker_dominance': berger_parker,
                    
                    # Hill numbers (True Diversity)
                    'hill_0_richness': hill_0,
                    'hill_1_shannon_exp': hill_1,
                    'hill_2_simpson_inv': hill_2,
                    
                    # Evenness measures
                    'evenness_pielou': evenness,
                    'evenness_shannon': shannon_evenness,
                    
                    # Advanced metrics
                    'renyi_2_diversity': renyi_2,
                    'theil_diversity_index': theil_index,
                    'effective_groups': effective_groups,
                    
                    # Descriptive stats
                    'total_groups': len(proportions),
                    'dominant_group_proportion': berger_parker,
                    'minority_groups_count': sum(1 for p in proportions if p < 0.1),
                    
                    # Overall scores
                    'diversity_score': self._calculate_overall_diversity_score(shannon_diversity, evenness, berger_parker),
                    'diversity_level': diversity_level,
                    
                    # Tracking metadata
                    'calculation_timestamp': pd.Timestamp.now().isoformat(),
                    'sample_size': len(self.dataset)
                }
        
        return metrics
    
    def _calculate_overall_diversity_score(self, shannon: float, evenness: float, dominance: float) -> float:
        """Calculate overall diversity score (0-1 scale, higher is more diverse)"""
        # Normalize Shannon diversity (typical range 0-3)
        normalized_shannon = min(shannon / 3.0, 1.0)
        
        # Evenness is already 0-1
        # Convert dominance to diversity (lower dominance = higher diversity)
        diversity_from_dominance = 1 - dominance
        
        # Weighted average
        return (0.4 * normalized_shannon + 0.3 * evenness + 0.3 * diversity_from_dominance)
    
    def _classify_diversity_level(self, shannon: float, num_groups: int) -> str:
        """Classify diversity level based on Shannon index and group count"""
        if num_groups == 1:
            return "No Diversity (Monolithic)"
        elif shannon < 0.5:
            return "Very Low Diversity"
        elif shannon < 1.0:
            return "Low Diversity"
        elif shannon < 1.5:
            return "Moderate Diversity"
        elif shannon < 2.0:
            return "High Diversity"
        else:
            return "Very High Diversity"
    
    def track_diversity_over_time(self, time_column: str = None, time_windows: List[str] = None) -> Dict[str, Any]:
        """Track diversity metrics over time periods"""
        if self.dataset is None:
            return {"error": "No dataset loaded"}
        
        tracking_results = {}
        
        # If no time column specified, create artificial time windows based on row indices
        if time_column is None or time_column not in self.dataset.columns:
            # Split data into temporal chunks for analysis
            n_chunks = 5  # Default 5 time periods
            chunk_size = len(self.dataset) // n_chunks
            
            for i, attr in enumerate(self.protected_attributes):
                if attr in self.dataset.columns:
                    temporal_metrics = []
                    
                    for chunk_idx in range(n_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = start_idx + chunk_size if chunk_idx < n_chunks - 1 else len(self.dataset)
                        
                        chunk_data = self.dataset.iloc[start_idx:end_idx]
                        proportions = chunk_data[attr].value_counts(normalize=True)
                        
                        shannon = -sum(p * np.log(p) for p in proportions if p > 0)
                        simpson = 1 - sum(p**2 for p in proportions)
                        
                        temporal_metrics.append({
                            'period': f"Period_{chunk_idx + 1}",
                            'sample_size': len(chunk_data),
                            'shannon_diversity': shannon,
                            'simpson_diversity': simpson,
                            'num_groups': len(proportions),
                            'dominant_proportion': proportions.max()
                        })
                    
                    # Calculate trends
                    shannon_values = [m['shannon_diversity'] for m in temporal_metrics]
                    shannon_trend = 'Increasing' if shannon_values[-1] > shannon_values[0] else 'Decreasing' if shannon_values[-1] < shannon_values[0] else 'Stable'
                    
                    tracking_results[attr] = {
                        'temporal_metrics': temporal_metrics,
                        'shannon_trend': shannon_trend,
                        'diversity_volatility': np.std(shannon_values),
                        'trend_strength': abs(shannon_values[-1] - shannon_values[0])
                    }
        
        return tracking_results
    
    def compare_diversity_across_subgroups(self, groupby_column: str) -> Dict[str, Any]:
        """Compare diversity metrics across different subgroups"""
        if self.dataset is None or groupby_column not in self.dataset.columns:
            return {"error": "Dataset not loaded or groupby column not found"}
        
        subgroup_analysis = {}
        
        for subgroup_value in self.dataset[groupby_column].unique():
            subgroup_data = self.dataset[self.dataset[groupby_column] == subgroup_value]
            
            # Temporarily set subgroup data and calculate metrics
            original_dataset = self.dataset
            self.dataset = subgroup_data
            
            subgroup_metrics = self.calculate_diversity_metrics()
            
            # Restore original dataset
            self.dataset = original_dataset
            
            subgroup_analysis[str(subgroup_value)] = {
                'sample_size': len(subgroup_data),
                'diversity_metrics': subgroup_metrics
            }
        
        return {
            'groupby_column': groupby_column,
            'subgroup_analysis': subgroup_analysis,
            'comparison_summary': self._summarize_subgroup_diversity_differences(subgroup_analysis)
        }
    
    def _summarize_subgroup_diversity_differences(self, subgroup_analysis: Dict) -> Dict[str, Any]:
        """Summarize diversity differences across subgroups"""
        summary = {}
        
        for attr in self.protected_attributes:
            if attr in ['error']:
                continue
                
            diversity_scores = []
            subgroup_names = []
            
            for subgroup_name, subgroup_data in subgroup_analysis.items():
                if attr in subgroup_data.get('diversity_metrics', {}):
                    score = subgroup_data['diversity_metrics'][attr].get('diversity_score', 0)
                    diversity_scores.append(score)
                    subgroup_names.append(subgroup_name)
            
            if diversity_scores:
                max_idx = np.argmax(diversity_scores)
                min_idx = np.argmin(diversity_scores)
                
                summary[attr] = {
                    'most_diverse_subgroup': subgroup_names[max_idx],
                    'least_diverse_subgroup': subgroup_names[min_idx],
                    'diversity_range': max(diversity_scores) - min(diversity_scores),
                    'average_diversity': np.mean(diversity_scores),
                    'diversity_std': np.std(diversity_scores)
                }
        
        return summary
    
    def save_diversity_tracking_report(self, filepath: str = "reports/diversity_tracking.json") -> bool:
        """Save comprehensive diversity tracking report"""
        try:
            # Generate all diversity analyses
            basic_metrics = self.calculate_diversity_metrics()
            temporal_tracking = self.track_diversity_over_time()
            
            # Try subgroup analysis if possible (e.g., by age groups)
            subgroup_comparison = {}
            potential_groupby_cols = ['age', 'education', 'workclass', 'occupation']
            
            for col in potential_groupby_cols:
                if col in self.dataset.columns:
                    try:
                        subgroup_comparison[col] = self.compare_diversity_across_subgroups(col)
                        break  # Use first available grouping column
                    except:
                        continue
            
            # Compile comprehensive report
            tracking_report = {
                'report_metadata': {
                    'dataset_size': len(self.dataset),
                    'protected_attributes': self.protected_attributes,
                    'generation_timestamp': pd.Timestamp.now().isoformat(),
                    'report_type': 'Diversity Tracking Analysis'
                },
                'current_diversity_metrics': basic_metrics,
                'temporal_diversity_tracking': temporal_tracking,
                'subgroup_diversity_comparison': subgroup_comparison,
                'diversity_benchmarking': self._benchmark_diversity_metrics(basic_metrics),
                'recommendations': self._generate_diversity_recommendations(basic_metrics, temporal_tracking)
            }
            
            # Save to file
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            import json
            with open(filepath, 'w') as f:
                json.dump(tracking_report, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error saving diversity tracking report: {e}")
            return False
    
    def _benchmark_diversity_metrics(self, metrics: Dict) -> Dict[str, Any]:
        """Benchmark diversity metrics against ideal standards"""
        benchmarks = {}
        
        for attr, attr_metrics in metrics.items():
            if isinstance(attr_metrics, dict):
                shannon = attr_metrics.get('shannon_diversity', 0)
                evenness = attr_metrics.get('evenness_pielou', 0)
                num_groups = attr_metrics.get('total_groups', 0)
                
                # Theoretical maximum Shannon diversity for this number of groups
                max_shannon = np.log(num_groups) if num_groups > 0 else 0
                shannon_efficiency = shannon / max_shannon if max_shannon > 0 else 0
                
                benchmarks[attr] = {
                    'shannon_efficiency': shannon_efficiency,
                    'evenness_rating': 'Excellent' if evenness > 0.8 else 'Good' if evenness > 0.6 else 'Fair' if evenness > 0.4 else 'Poor',
                    'diversity_adequacy': 'Sufficient' if shannon > 1.5 else 'Moderate' if shannon > 1.0 else 'Low',
                    'benchmark_score': (shannon_efficiency + evenness) / 2
                }
        
        return benchmarks
    
    def _generate_diversity_recommendations(self, current_metrics: Dict, temporal_data: Dict) -> List[str]:
        """Generate actionable diversity improvement recommendations"""
        recommendations = []
        
        for attr, metrics in current_metrics.items():
            if isinstance(metrics, dict):
                diversity_score = metrics.get('diversity_score', 0)
                shannon = metrics.get('shannon_diversity', 0)
                evenness = metrics.get('evenness_pielou', 0)
                dominance = metrics.get('berger_parker_dominance', 0)
                
                attr_display = attr.replace('_', ' ').title()
                
                if diversity_score < 0.4:
                    recommendations.append(f"CRITICAL: {attr_display} shows very low diversity (score: {diversity_score:.2f}). Consider targeted recruitment of underrepresented groups.")
                
                if dominance > 0.7:
                    recommendations.append(f"WARNING: {attr_display} is dominated by one group ({dominance:.1%}). Implement diversity initiatives to balance representation.")
                
                if evenness < 0.5:
                    recommendations.append(f"IMPROVE: {attr_display} has uneven distribution. Focus on increasing representation of smaller groups.")
                
                if shannon < 1.0:
                    recommendations.append(f"ENHANCE: {attr_display} diversity could be improved (Shannon: {shannon:.2f}). Target diversity score above 1.5.")
        
        # Temporal recommendations
        for attr, temporal_metrics in temporal_data.items():
            if temporal_metrics.get('shannon_trend') == 'Decreasing':
                recommendations.append(f"TREND ALERT: {attr.replace('_', ' ').title()} diversity is decreasing over time. Investigate causes and implement corrective measures.")
        
        # General recommendations
        if not recommendations:
            recommendations.append("MAINTAIN: Current diversity levels are adequate. Continue monitoring and maintain inclusive practices.")
        
        recommendations.extend([
            "MONITOR: Establish regular diversity tracking with quarterly assessments.",
            "BENCHMARK: Compare diversity metrics against industry standards and best practices.",
            "DOCUMENT: Maintain diversity reports for compliance and continuous improvement."
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def evaluate_historical_context(self) -> Dict[str, Any]:
        """Evaluate representation in historical context"""
        context = {
            'dataset_era': 'Adult dataset (1994 US Census)',
            'historical_notes': {
                'gender': ' 1994 binary gender classification standard; modern non-binary identities not captured',
                'race': 'US Census racial categories circa 1994; Hispanic/Latino treated as separate ethnicity',
                'age': 'Adult population only (16+); excludes youth demographics',
                'geography': 'US-centric; limited international representation'
            },
            'modern_relevance': {
                'strengths': [
                    'Large sample size for statistical validity',
                    'Multiple demographic dimensions captured',
                    'Economic outcome focus relevant across time periods'
                ],
                'limitations': [
                    'Outdated demographic compositions',
                    'Limited identity categories by modern standards',
                    'Geographic bias toward US populations',
                    'Binary gender assumption'
                ]
            },
            'recommendations': [
                'Supplement with modern demographic data',
                'Consider expanded identity categories',
                'Weight samples to match current population',
                'Include international diversity benchmarks'
            ]
        }
        
        return context
    
    def generate_representation_report(self) -> Dict[str, Any]:
        """Generate comprehensive representation bias report"""
        if self.dataset is None:
            return {"error": "No dataset loaded"}
        
        report = {
            'dataset_info': {
                'total_samples': len(self.dataset),
                'attributes_analyzed': self.protected_attributes,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },
            'demographic_distribution': self.analyze_demographic_distribution(),
            'representation_gaps': self.quantify_representation_gaps(),
            'inclusion_gaps': self.identify_inclusion_gaps(),
            'diversity_metrics': self.calculate_diversity_metrics(),
            'historical_context': self.evaluate_historical_context(),
            'overall_assessment': self._generate_overall_assessment()
        }
        
        return report
    
    def _generate_overall_assessment(self) -> Dict[str, Any]:
        """Generate overall representation bias assessment"""
        diversity_metrics = self.calculate_diversity_metrics()
        representation_gaps = self.quantify_representation_gaps()
        inclusion_gaps = self.identify_inclusion_gaps()
        
        # Calculate overall scores
        diversity_scores = []
        representation_scores = []
        inclusion_scores = []
        
        for attr in self.protected_attributes:
            if attr in diversity_metrics:
                diversity_scores.append(diversity_metrics[attr]['diversity_score'])
            
            if attr in representation_gaps:
                # Calculate representation score based on how many groups are adequately represented
                adequate_count = sum(1 for group_data in representation_gaps[attr].values() 
                                   if group_data['status'] in ['Adequately Represented', 'Slightly Over-represented'])
                total_groups = len(representation_gaps[attr])
                representation_scores.append(adequate_count / total_groups if total_groups > 0 else 0)
            
            if attr in inclusion_gaps:
                inclusion_scores.append(inclusion_gaps[attr]['inclusion_score'])
        
        overall_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
        overall_representation = np.mean(representation_scores) if representation_scores else 0.0
        overall_inclusion = np.mean(inclusion_scores) if inclusion_scores else 0.0
        
        # Combined bias score (higher = less biased)
        combined_score = (overall_diversity + overall_representation + overall_inclusion) / 3
        
        # Determine bias level
        if combined_score >= 0.8:
            bias_level = "Low Representation Bias"
        elif combined_score >= 0.6:
            bias_level = "Moderate Representation Bias"
        elif combined_score >= 0.4:
            bias_level = "High Representation Bias"
        else:
            bias_level = "Severe Representation Bias"
        
        return {
            'overall_diversity_score': overall_diversity,
            'overall_representation_score': overall_representation,
            'overall_inclusion_score': overall_inclusion,
            'combined_bias_score': combined_score,
            'bias_level': bias_level,
            'key_concerns': self._identify_key_concerns(),
            'priority_actions': self._recommend_priority_actions()
        }
    
    def _identify_key_concerns(self) -> List[str]:
        """Identify key representation concerns"""
        concerns = []
        
        gaps = self.quantify_representation_gaps()
        for attr, attr_gaps in gaps.items():
            for group, group_data in attr_gaps.items():
                if group_data['status'] in ['Severely Under-represented', 'Under-represented']:
                    concerns.append(f"{group} in {attr} is {group_data['status'].lower()}")
                elif group_data['status'] == 'Over-represented':
                    concerns.append(f"{group} in {attr} is over-represented")
        
        diversity = self.calculate_diversity_metrics()
        for attr, metrics in diversity.items():
            if metrics['diversity_score'] < 0.5:
                concerns.append(f"Low diversity in {attr} (score: {metrics['diversity_score']:.2f})")
        
        return concerns[:10]  # Limit to top 10 concerns
    
    def _recommend_priority_actions(self) -> List[str]:
        """Recommend priority actions to address representation bias"""
        actions = [
            "Conduct targeted data collection for under-represented groups",
            "Implement stratified sampling to ensure demographic balance",
            "Consider data augmentation techniques for minority groups",
            "Apply post-processing weights to correct representation imbalances",
            "Establish ongoing monitoring of demographic representation",
            "Expand identity categories to capture modern diversity",
            "Validate findings with multiple demographic data sources",
            "Document representation limitations in model cards"
        ]
        
        return actions
    
    def visualize_representation(self, save_path: str = None) -> plt.Figure:
        """Create visualization of representation analysis"""
        if self.dataset is None:
            print("No dataset loaded")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Representation Bias Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Demographic distribution
        if len(self.protected_attributes) > 0:
            attr = self.protected_attributes[0]
            if attr in self.dataset.columns:
                self.dataset[attr].value_counts().plot(kind='bar', ax=axes[0,0])
                axes[0,0].set_title(f'{attr.title()} Distribution')
                axes[0,0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Representation gaps (if benchmarks available)
        gaps = self.quantify_representation_gaps()
        if gaps:
            attr = list(gaps.keys())[0]
            groups = list(gaps[attr].keys())
            ratios = [gaps[attr][group]['representation_ratio'] for group in groups]
            
            colors = ['red' if r < 0.85 else 'orange' if r < 1.15 else 'green' for r in ratios]
            axes[0,1].bar(groups, ratios, color=colors, alpha=0.7)
            axes[0,1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            axes[0,1].set_title(f'{attr.title()} Representation Ratios')
            axes[0,1].set_ylabel('Observed/Expected Ratio')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Diversity scores
        diversity = self.calculate_diversity_metrics()
        if diversity:
            attrs = list(diversity.keys())
            scores = [diversity[attr]['diversity_score'] for attr in attrs]
            axes[1,0].bar(attrs, scores, color='skyblue', alpha=0.7)
            axes[1,0].set_title('Diversity Scores by Attribute')
            axes[1,0].set_ylabel('Diversity Score (0-1)')
            axes[1,0].set_ylim(0, 1)
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Overall assessment
        assessment = self._generate_overall_assessment()
        metrics = ['Diversity', 'Representation', 'Inclusion']
        values = [
            assessment['overall_diversity_score'],
            assessment['overall_representation_score'], 
            assessment['overall_inclusion_score']
        ]
        
        colors = ['red' if v < 0.4 else 'orange' if v < 0.7 else 'green' for v in values]
        axes[1,1].bar(metrics, values, color=colors, alpha=0.7)
        axes[1,1].set_title('Overall Assessment Scores')
        axes[1,1].set_ylabel('Score (0-1)')
        axes[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def main():
    """Example usage of RepresentationBiasAgent"""
    agent = RepresentationBiasAgent()
    
    # Load dataset
    if agent.load_dataset("data/adult.csv", ["gender", "race"]):
        print("Dataset loaded successfully")
        
        # Generate comprehensive report
        report = agent.generate_representation_report()
        
        # Print key findings
        print("\n=== REPRESENTATION BIAS ANALYSIS ===")
        print(f"Dataset size: {report['dataset_info']['total_samples']} samples")
        print(f"Analyzed attributes: {report['dataset_info']['attributes_analyzed']}")
        
        assessment = report['overall_assessment']
        print(f"\nOverall Bias Level: {assessment['bias_level']}")
        print(f"Combined Bias Score: {assessment['combined_bias_score']:.3f}")
        
        print("\nKey Concerns:")
        for concern in assessment['key_concerns'][:5]:
            print(f"  • {concern}")
        
        # Create visualization
        agent.visualize_representation("reports/representation_analysis.png")
        print("\nVisualization saved to reports/representation_analysis.png")


if __name__ == "__main__":
    main()

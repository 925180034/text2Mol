#!/usr/bin/env python3
"""
Optimized Prompting Strategies for MolT5-Large Caption2SMILES Model.
Implements sophisticated prompting templates and strategies specifically designed
for the laituan245/molt5-large-caption2smiles model to maximize SMILES generation quality.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import yaml
import logging
import time
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedMolecularPrompting:
    """
    Advanced prompting system specifically optimized for MolT5-Large Caption2SMILES.
    Uses evidence-based prompt engineering and molecular domain knowledge.
    """
    
    def __init__(self, model_path: str = "models/MolT5-Large-Caption2SMILES"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        logger.info(f"🚀 Loading MolT5-Large Caption2SMILES from {model_path}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ Model loaded on {self.device}")
        logger.info(f"📊 Parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M")
        
        # Initialize prompt templates
        self.prompt_templates = self._initialize_prompt_templates()
        
        # Performance tracking
        self.performance_stats = {
            'template_success_rates': {},
            'molecular_class_performance': {},
            'total_generations': 0,
            'valid_generations': 0
        }

    def _initialize_prompt_templates(self) -> Dict[str, List[str]]:
        """
        Initialize optimized prompt templates based on the caption2smiles training.
        These templates are designed to match the training data format of the model.
        """
        
        templates = {
            # Direct Caption Style (matches training data format)
            'caption_direct': [
                "{description}",
                "molecular structure: {description}",
                "compound: {description}",
                "chemical: {description}",
                "molecule: {description}"
            ],
            
            # Question-Answer Format
            'question_answer': [
                "What is the SMILES of {description}?",
                "Generate SMILES for {description}",
                "SMILES representation of {description}",
                "Convert to SMILES: {description}",
                "Molecular SMILES for {description}"
            ],
            
            # Template-Based Format
            'template_based': [
                "Description: {description}\nSMILES:",
                "Compound description: {description}\nMolecular formula:",
                "Chemical name: {description}\nSMILES notation:",
                "Molecule: {description}\nStructure:",
                "Chemical structure for {description}:"
            ],
            
            # Property-Enhanced Format
            'property_enhanced': [
                "Generate SMILES for {description} (molecular compound)",
                "Chemical structure of {description} as SMILES",
                "Organic molecule {description} in SMILES format",
                "Pharmaceutical compound {description} to SMILES",
                "Bioactive molecule {description} as SMILES"
            ],
            
            # Context-Rich Format
            'context_rich': [
                "In chemistry, {description} has the SMILES structure:",
                "The molecular representation of {description} is:",
                "Chemical database entry for {description}:",
                "Molecular formula for compound {description}:",
                "SMILES notation for chemical {description}:"
            ],
            
            # Few-Shot Style Templates
            'few_shot': [
                "Examples:\nwater -> O\nmethane -> C\n{description} ->",
                "Chemical to SMILES conversion:\nExample: ethanol -> CCO\nTarget: {description} ->",
                "Molecule structure mapping:\nwater: O\nbenzene: c1ccccc1\n{description}:",
                "SMILES generation:\nInput: water, Output: O\nInput: {description}, Output:",
                "Caption to SMILES:\nmethane -> C\nammonia -> N\n{description} ->"
            ]
        }
        
        return templates

    def _preprocess_description(self, description: str) -> str:
        """
        Preprocess molecular descriptions for optimal model performance.
        """
        # Clean and normalize description
        description = description.strip().lower()
        
        # Handle common molecular naming patterns
        replacements = {
            'molecule': '',
            'compound': '',
            'chemical': '',
            'structure': '',
            'formula': '',
            'substance': ''
        }
        
        for old, new in replacements.items():
            description = description.replace(old, new).strip()
        
        # Remove extra whitespace
        description = ' '.join(description.split())
        
        return description

    def _select_optimal_template(self, description: str, molecular_class: str = None) -> Tuple[str, str]:
        """
        Select optimal prompt template based on description characteristics and performance history.
        """
        # Analyze description characteristics
        desc_lower = description.lower()
        
        # Template selection logic based on description type
        if any(term in desc_lower for term in ['water', 'methane', 'ethanol', 'benzene', 'simple']):
            # Use direct caption for simple, well-known molecules
            template_category = 'caption_direct'
        elif any(term in desc_lower for term in ['drug', 'pharmaceutical', 'medicine', 'therapeutic']):
            # Use property-enhanced for pharmaceutical compounds
            template_category = 'property_enhanced'
        elif any(term in desc_lower for term in ['organic', 'aromatic', 'aliphatic', 'cyclic']):
            # Use context-rich for organic chemistry terms
            template_category = 'context_rich'
        elif len(description.split()) <= 3:
            # Use few-shot for very short descriptions
            template_category = 'few_shot'
        else:
            # Default to question-answer format
            template_category = 'question_answer'
        
        # Select best performing template from category
        templates = self.prompt_templates[template_category]
        
        # Use performance history if available
        if template_category in self.performance_stats['template_success_rates']:
            rates = self.performance_stats['template_success_rates'][template_category]
            if rates:
                # Select template with highest success rate
                best_idx = max(range(len(rates)), key=lambda i: rates.get(i, 0))
                template = templates[min(best_idx, len(templates)-1)]
            else:
                # Use first template if no history
                template = templates[0]
        else:
            # Use first template if no history
            template = templates[0]
        
        return template_category, template

    def generate_smiles(self, description: str, 
                       num_candidates: int = 5,
                       temperature: float = 0.7,
                       use_ensemble: bool = True) -> Dict:
        """
        Generate SMILES using optimized prompting strategies.
        """
        # Preprocess description
        processed_desc = self._preprocess_description(description)
        
        # Select optimal template
        template_category, template = self._select_optimal_template(processed_desc)
        
        results = {
            'input_description': description,
            'processed_description': processed_desc,
            'template_category': template_category,
            'template_used': template,
            'candidates': [],
            'best_candidate': None,
            'validity_scores': [],
            'generation_success': False
        }
        
        if use_ensemble:
            # Generate with multiple templates for ensemble approach
            candidates = self._generate_ensemble_candidates(processed_desc, num_candidates, temperature)
        else:
            # Generate with single optimal template
            candidates = self._generate_single_template_candidates(
                processed_desc, template, num_candidates, temperature
            )
        
        # Evaluate and rank candidates
        ranked_candidates = self._evaluate_and_rank_candidates(candidates, processed_desc)
        
        results['candidates'] = ranked_candidates
        if ranked_candidates:
            results['best_candidate'] = ranked_candidates[0]
            results['generation_success'] = ranked_candidates[0]['is_valid']
        
        # Update performance statistics
        self._update_performance_stats(template_category, results['generation_success'])
        
        return results

    def _generate_ensemble_candidates(self, description: str, num_candidates: int, temperature: float) -> List[Dict]:
        """Generate candidates using ensemble of different templates."""
        candidates = []
        
        # Use top 3 template categories
        top_categories = ['caption_direct', 'question_answer', 'few_shot']
        
        for category in top_categories:
            templates = self.prompt_templates[category]
            
            # Use best template from each category
            template = templates[0]  # Could be improved with performance tracking
            
            # Generate multiple candidates with this template
            candidates_per_template = max(1, num_candidates // len(top_categories))
            
            template_candidates = self._generate_single_template_candidates(
                description, template, candidates_per_template, temperature
            )
            
            for candidate in template_candidates:
                candidate['template_category'] = category
                candidate['template'] = template
            
            candidates.extend(template_candidates)
        
        return candidates[:num_candidates]

    def _generate_single_template_candidates(self, description: str, template: str, 
                                           num_candidates: int, temperature: float) -> List[Dict]:
        """Generate candidates using a single template."""
        candidates = []
        
        # Format prompt
        prompt = template.format(description=description)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding='max_length'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate multiple candidates
        with torch.no_grad():
            for i in range(num_candidates):
                # Use different sampling strategies
                if i == 0:
                    # Greedy decoding for most likely result
                    outputs = self.model.generate(
                        **inputs,
                        max_length=128,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                elif i == 1:
                    # Beam search for structured exploration
                    outputs = self.model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=5,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                else:
                    # Sampling with different temperatures
                    sample_temp = temperature * (0.5 + 0.5 * i / num_candidates)
                    outputs = self.model.generate(
                        **inputs,
                        max_length=128,
                        do_sample=True,
                        temperature=sample_temp,
                        top_k=50,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode result
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Clean generated text
                cleaned_smiles = self._clean_generated_smiles(generated_text)
                
                candidates.append({
                    'smiles': cleaned_smiles,
                    'raw_output': generated_text,
                    'generation_method': ['greedy', 'beam', 'sample'][min(i, 2)],
                    'temperature': temperature if i >= 2 else None
                })
        
        return candidates

    def _clean_generated_smiles(self, generated_text: str) -> str:
        """Clean and extract SMILES from generated text."""
        # Remove common prefixes/suffixes
        text = generated_text.strip()
        
        # Remove common non-SMILES patterns
        patterns_to_remove = [
            'SMILES:', 'smiles:', 'Structure:', 'structure:',
            'Formula:', 'formula:', 'Notation:', 'notation:',
            'Representation:', 'representation:'
        ]
        
        for pattern in patterns_to_remove:
            text = text.replace(pattern, '').strip()
        
        # Extract potential SMILES (usually the first word/continuous string)
        words = text.split()
        if words:
            # Look for the most SMILES-like word
            for word in words:
                if self._looks_like_smiles(word):
                    return word
            # If no obvious SMILES, return first word
            return words[0]
        
        return text

    def _looks_like_smiles(self, text: str) -> bool:
        """Check if text looks like a SMILES string."""
        smiles_chars = set('CNOPSFClBrI()[]=#+-123456789@/\\cno')
        text_chars = set(text)
        
        # Check if majority of characters are SMILES-like
        smiles_ratio = len(text_chars.intersection(smiles_chars)) / max(len(text_chars), 1)
        
        return smiles_ratio > 0.7 and len(text) > 1

    def _evaluate_and_rank_candidates(self, candidates: List[Dict], description: str) -> List[Dict]:
        """Evaluate and rank candidates based on validity and quality metrics."""
        
        for candidate in candidates:
            smiles = candidate['smiles']
            
            # Basic validity check
            mol = Chem.MolFromSmiles(smiles)
            candidate['is_valid'] = mol is not None
            
            if mol is not None:
                # Calculate molecular properties
                try:
                    candidate['molecular_weight'] = Descriptors.MolWt(mol)
                    candidate['logp'] = Descriptors.MolLogP(mol)
                    candidate['num_atoms'] = mol.GetNumAtoms()
                    candidate['num_bonds'] = mol.GetNumBonds()
                    candidate['num_rings'] = rdMolDescriptors.CalcNumRings(mol)
                    
                    # Quality score based on properties
                    quality_score = 1.0
                    
                    # Prefer reasonable molecular weights
                    if 50 <= candidate['molecular_weight'] <= 800:
                        quality_score += 0.2
                    
                    # Prefer reasonable LogP values
                    if -5 <= candidate['logp'] <= 5:
                        quality_score += 0.2
                    
                    # Prefer structures with reasonable complexity
                    if 3 <= candidate['num_atoms'] <= 50:
                        quality_score += 0.2
                        
                    candidate['quality_score'] = quality_score
                    
                except Exception as e:
                    logger.warning(f"Error calculating properties for {smiles}: {e}")
                    candidate['quality_score'] = 0.5
            else:
                candidate['quality_score'] = 0.0
        
        # Sort by validity first, then by quality score
        ranked_candidates = sorted(
            candidates,
            key=lambda x: (x['is_valid'], x['quality_score']),
            reverse=True
        )
        
        return ranked_candidates

    def _update_performance_stats(self, template_category: str, success: bool):
        """Update performance statistics for continuous improvement."""
        self.performance_stats['total_generations'] += 1
        if success:
            self.performance_stats['valid_generations'] += 1
        
        # Update template-specific stats
        if template_category not in self.performance_stats['template_success_rates']:
            self.performance_stats['template_success_rates'][template_category] = {}
        
        category_stats = self.performance_stats['template_success_rates'][template_category]
        category_stats.setdefault('total', 0)
        category_stats.setdefault('success', 0)
        
        category_stats['total'] += 1
        if success:
            category_stats['success'] += 1

    def batch_generate(self, descriptions: List[str], **kwargs) -> List[Dict]:
        """Generate SMILES for multiple descriptions."""
        results = []
        
        logger.info(f"🧪 Batch generating SMILES for {len(descriptions)} descriptions")
        
        for i, desc in enumerate(descriptions):
            logger.info(f"Processing {i+1}/{len(descriptions)}: {desc[:50]}...")
            result = self.generate_smiles(desc, **kwargs)
            results.append(result)
        
        return results

    def get_performance_report(self) -> Dict:
        """Get detailed performance report."""
        total = self.performance_stats['total_generations']
        valid = self.performance_stats['valid_generations']
        
        report = {
            'overall_success_rate': valid / max(total, 1) * 100,
            'total_generations': total,
            'valid_generations': valid,
            'template_performance': {}
        }
        
        # Template-specific performance
        for category, stats in self.performance_stats['template_success_rates'].items():
            if 'total' in stats and stats['total'] > 0:
                success_rate = stats['success'] / stats['total'] * 100
                report['template_performance'][category] = {
                    'success_rate': success_rate,
                    'total_uses': stats['total'],
                    'successful_uses': stats['success']
                }
        
        return report

def test_optimized_prompting():
    """Test the optimized prompting system."""
    
    logger.info("🧪 TESTING OPTIMIZED PROMPTING SYSTEM")
    logger.info("=" * 60)
    
    # Initialize prompting system
    prompting = OptimizedMolecularPrompting()
    
    # Test cases covering different types of molecular descriptions
    test_cases = [
        # Simple molecules
        "water",
        "methane",
        "ethanol",
        "benzene",
        
        # Organic compounds
        "acetic acid",
        "glucose",
        "caffeine",
        "aspirin",
        
        # Pharmaceutical compounds
        "ibuprofen",
        "acetaminophen",
        "morphine",
        "penicillin",
        
        # Chemical classes
        "simple alcohol",
        "aromatic compound with six carbons",
        "carboxylic acid with two carbons",
        "cyclic hydrocarbon with five carbons"
    ]
    
    # Generate SMILES for each test case
    results = []
    
    for i, description in enumerate(test_cases):
        logger.info(f"\n🔬 Test {i+1}/{len(test_cases)}: {description}")
        
        # Generate with ensemble approach
        result = prompting.generate_smiles(
            description,
            num_candidates=5,
            temperature=0.7,
            use_ensemble=True
        )
        
        results.append(result)
        
        # Display results
        if result['best_candidate']:
            best = result['best_candidate']
            validity = "✅" if best['is_valid'] else "❌"
            logger.info(f"  Best: {best['smiles']} {validity}")
            logger.info(f"  Template: {result['template_category']}")
            
            if best['is_valid']:
                logger.info(f"  MW: {best.get('molecular_weight', 'N/A'):.1f}, "
                          f"LogP: {best.get('logp', 'N/A'):.2f}")
        else:
            logger.info("  ❌ No valid candidates generated")
    
    # Performance summary
    logger.info(f"\n📊 PERFORMANCE SUMMARY")
    logger.info("=" * 40)
    
    report = prompting.get_performance_report()
    logger.info(f"Overall success rate: {report['overall_success_rate']:.1f}%")
    logger.info(f"Total generations: {report['total_generations']}")
    logger.info(f"Valid generations: {report['valid_generations']}")
    
    # Template performance
    logger.info(f"\n📋 Template Performance:")
    for category, stats in report['template_performance'].items():
        logger.info(f"  {category}: {stats['success_rate']:.1f}% "
                   f"({stats['successful_uses']}/{stats['total_uses']})")
    
    return results, report

def create_molecular_prompt_templates():
    """Create comprehensive prompt templates for different molecular generation tasks."""
    
    templates = {
        'text_to_smiles': {
            'caption_style': [
                "{description}",
                "molecular structure: {description}",
                "compound: {description}",
                "chemical: {description}"
            ],
            'instruction_style': [
                "Generate SMILES for {description}",
                "Convert to SMILES: {description}",
                "What is the SMILES of {description}?",
                "SMILES representation of {description}"
            ],
            'template_style': [
                "Description: {description}\nSMILES:",
                "Compound: {description}\nStructure:",
                "Chemical name: {description}\nSMILES notation:",
                "Molecule: {description}\nFormula:"
            ]
        },
        
        'smiles_to_properties': {
            'property_prediction': [
                "SMILES: {smiles}\nMolecular weight:",
                "Structure: {smiles}\nLogP value:",  
                "Compound: {smiles}\nSolubility:",
                "Molecule: {smiles}\nBioactivity:"
            ]
        },
        
        'scaffold_conditioning': [
            "Generate molecule with scaffold {scaffold} and properties: {description}",
            "Create SMILES containing substructure {scaffold} for: {description}",
            "Design molecule with core {scaffold} having: {description}",
            "Scaffold: {scaffold}\nTarget properties: {description}\nSMILES:"
        ]
    }
    
    return templates

def main():
    """Main testing and demonstration function."""
    
    try:
        # Test optimized prompting system
        results, report = test_optimized_prompting()
        
        # Save results
        output_dir = Path("outputs/prompting_optimization")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(output_dir / "prompting_test_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save performance report
        with open(output_dir / "performance_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n🎉 Optimized prompting system tested successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
        return results, report
        
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == '__main__':
    results, report = main()
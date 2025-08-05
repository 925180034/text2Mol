#!/usr/bin/env python3
"""
简化版评估指标实现 - 包含所有9个指标
"""

import numpy as np
from typing import List, Dict, Set
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.DataStructs import TanimotoSimilarity
import Levenshtein
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 下载NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class SimpleMetrics:
    """实现所有9个评估指标的简化版本"""
    
    def __init__(self):
        self.smoothing = SmoothingFunction()
    
    def validity(self, smiles_list: List[str]) -> float:
        """
        指标1: 有效性 - 生成的SMILES是否为有效分子
        """
        if not smiles_list:
            return 0.0
        
        valid_count = 0
        for smiles in smiles_list:
            if smiles and smiles.strip():
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        valid_count += 1
                except:
                    pass
        
        return valid_count / len(smiles_list)
    
    def uniqueness(self, smiles_list: List[str]) -> float:
        """
        指标2: 唯一性 - 生成分子的去重比例
        """
        if not smiles_list:
            return 0.0
        
        # 只考虑有效的SMILES
        valid_smiles = []
        for smiles in smiles_list:
            if smiles and smiles.strip():
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        # 标准化SMILES
                        canonical_smiles = Chem.MolToSmiles(mol)
                        valid_smiles.append(canonical_smiles)
                except:
                    pass
        
        if not valid_smiles:
            return 0.0
        
        unique_smiles = set(valid_smiles)
        return len(unique_smiles) / len(valid_smiles)
    
    def novelty(self, generated_smiles: List[str], reference_smiles: List[str]) -> float:
        """
        指标3: 新颖性 - 生成的分子不在参考集中的比例
        """
        if not generated_smiles:
            return 0.0
        
        # 获取有效的生成分子
        valid_generated = set()
        for smiles in generated_smiles:
            if smiles and smiles.strip():
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        canonical_smiles = Chem.MolToSmiles(mol)
                        valid_generated.add(canonical_smiles)
                except:
                    pass
        
        if not valid_generated:
            return 0.0
        
        # 获取参考集的标准化SMILES
        reference_set = set()
        for smiles in reference_smiles:
            if smiles and smiles.strip():
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        canonical_smiles = Chem.MolToSmiles(mol)
                        reference_set.add(canonical_smiles)
                except:
                    pass
        
        # 计算新颖分子
        novel_molecules = valid_generated - reference_set
        return len(novel_molecules) / len(valid_generated)
    
    def bleu_score(self, generated_smiles: List[str], target_smiles: List[str]) -> float:
        """
        指标4: BLEU分数 - 生成序列与目标序列的相似度
        """
        if not generated_smiles or not target_smiles:
            return 0.0
        
        if len(generated_smiles) != len(target_smiles):
            min_len = min(len(generated_smiles), len(target_smiles))
            generated_smiles = generated_smiles[:min_len]
            target_smiles = target_smiles[:min_len]
        
        scores = []
        for gen, tgt in zip(generated_smiles, target_smiles):
            if gen and tgt:
                # 将SMILES字符串转换为字符列表
                gen_tokens = list(gen)
                tgt_tokens = list(tgt)
                
                # 计算BLEU分数
                try:
                    score = sentence_bleu(
                        [tgt_tokens], 
                        gen_tokens,
                        smoothing_function=self.smoothing.method1
                    )
                    scores.append(score)
                except:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
    
    def exact_match_score(self, generated_smiles: List[str], target_smiles: List[str]) -> float:
        """
        指标5: 精确匹配 - 完全匹配目标分子的比例
        """
        if not generated_smiles or not target_smiles:
            return 0.0
        
        if len(generated_smiles) != len(target_smiles):
            min_len = min(len(generated_smiles), len(target_smiles))
            generated_smiles = generated_smiles[:min_len]
            target_smiles = target_smiles[:min_len]
        
        exact_matches = 0
        for gen, tgt in zip(generated_smiles, target_smiles):
            if gen and tgt:
                try:
                    # 标准化后比较
                    mol_gen = Chem.MolFromSmiles(gen)
                    mol_tgt = Chem.MolFromSmiles(tgt)
                    
                    if mol_gen is not None and mol_tgt is not None:
                        canonical_gen = Chem.MolToSmiles(mol_gen)
                        canonical_tgt = Chem.MolToSmiles(mol_tgt)
                        
                        if canonical_gen == canonical_tgt:
                            exact_matches += 1
                except:
                    pass
        
        return exact_matches / len(generated_smiles) if generated_smiles else 0.0
    
    def levenshtein_distance(self, generated_smiles: List[str], target_smiles: List[str]) -> float:
        """
        指标6: Levenshtein距离 - 编辑距离（归一化）
        """
        if not generated_smiles or not target_smiles:
            return 0.0
        
        if len(generated_smiles) != len(target_smiles):
            min_len = min(len(generated_smiles), len(target_smiles))
            generated_smiles = generated_smiles[:min_len]
            target_smiles = target_smiles[:min_len]
        
        distances = []
        for gen, tgt in zip(generated_smiles, target_smiles):
            if gen and tgt:
                # 计算编辑距离
                distance = Levenshtein.distance(gen, tgt)
                # 归一化（除以最大可能距离）
                max_len = max(len(gen), len(tgt))
                if max_len > 0:
                    normalized_distance = distance / max_len
                    # 转换为相似度分数（1 - 归一化距离）
                    similarity = 1 - normalized_distance
                    distances.append(similarity)
                else:
                    distances.append(1.0)
            else:
                distances.append(0.0)
        
        return np.mean(distances) if distances else 0.0
    
    def maccs_similarity(self, generated_smiles: List[str], target_smiles: List[str]) -> float:
        """
        指标7: MACCS指纹相似度
        """
        return self._compute_fingerprint_similarity(generated_smiles, target_smiles, 'maccs')
    
    def morgan_similarity(self, generated_smiles: List[str], target_smiles: List[str]) -> float:
        """
        指标8: Morgan指纹相似度
        """
        return self._compute_fingerprint_similarity(generated_smiles, target_smiles, 'morgan')
    
    def rdk_similarity(self, generated_smiles: List[str], target_smiles: List[str]) -> float:
        """
        指标9: RDKit指纹相似度
        """
        return self._compute_fingerprint_similarity(generated_smiles, target_smiles, 'rdk')
    
    def _compute_fingerprint_similarity(self, generated_smiles: List[str], 
                                      target_smiles: List[str], 
                                      fp_type: str) -> float:
        """
        计算分子指纹相似度
        """
        if not generated_smiles or not target_smiles:
            return 0.0
        
        if len(generated_smiles) != len(target_smiles):
            min_len = min(len(generated_smiles), len(target_smiles))
            generated_smiles = generated_smiles[:min_len]
            target_smiles = target_smiles[:min_len]
        
        similarities = []
        for gen, tgt in zip(generated_smiles, target_smiles):
            if gen and tgt:
                try:
                    mol_gen = Chem.MolFromSmiles(gen)
                    mol_tgt = Chem.MolFromSmiles(tgt)
                    
                    if mol_gen is not None and mol_tgt is not None:
                        # 生成指纹
                        if fp_type == 'maccs':
                            fp_gen = MACCSkeys.GenMACCSKeys(mol_gen)
                            fp_tgt = MACCSkeys.GenMACCSKeys(mol_tgt)
                        elif fp_type == 'morgan':
                            fp_gen = AllChem.GetMorganFingerprintAsBitVect(mol_gen, 2, nBits=2048)
                            fp_tgt = AllChem.GetMorganFingerprintAsBitVect(mol_tgt, 2, nBits=2048)
                        elif fp_type == 'rdk':
                            fp_gen = Chem.RDKFingerprint(mol_gen)
                            fp_tgt = Chem.RDKFingerprint(mol_tgt)
                        else:
                            continue
                        
                        # 计算Tanimoto相似度
                        similarity = TanimotoSimilarity(fp_gen, fp_tgt)
                        similarities.append(similarity)
                except Exception as e:
                    # 出错时添加0相似度
                    similarities.append(0.0)
            else:
                similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def calculate_fcd(self, generated_smiles: List[str], target_smiles: List[str]) -> float:
        """
        额外指标: FCD (Fréchet ChemNet Distance) - 简化版本
        注意：这是一个简化实现，真正的FCD需要预训练的ChemNet模型
        """
        # 简化版：基于分子属性的距离
        if not generated_smiles or not target_smiles:
            return float('inf')
        
        def get_mol_properties(smiles_list):
            """获取分子属性向量"""
            properties = []
            for smiles in smiles_list:
                if smiles:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            props = [
                                Descriptors.MolWt(mol),
                                Descriptors.MolLogP(mol),
                                Descriptors.NumHDonors(mol),
                                Descriptors.NumHAcceptors(mol),
                                Descriptors.TPSA(mol)
                            ]
                            properties.append(props)
                    except:
                        pass
            
            return np.array(properties) if properties else np.array([])
        
        gen_props = get_mol_properties(generated_smiles)
        tgt_props = get_mol_properties(target_smiles)
        
        if len(gen_props) == 0 or len(tgt_props) == 0:
            return float('inf')
        
        # 计算均值和协方差
        gen_mean = np.mean(gen_props, axis=0)
        tgt_mean = np.mean(tgt_props, axis=0)
        
        # 简化的Fréchet距离
        distance = np.linalg.norm(gen_mean - tgt_mean)
        
        return distance
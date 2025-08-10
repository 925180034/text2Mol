"""
简化的评估指标实现
包含10种主要的分子生成评估指标
"""

import numpy as np
from typing import List, Dict, Union
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import RDLogger
import Levenshtein
from collections import Counter

RDLogger.DisableLog('rdApp.*')


def calculate_validity(smiles_list: List[str]) -> float:
    """计算SMILES有效性"""
    if not smiles_list:
        return 0.0
    
    valid_count = 0
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_count += 1
        except:
            pass
    
    return valid_count / len(smiles_list)


def calculate_uniqueness(smiles_list: List[str]) -> float:
    """计算SMILES唯一性"""
    if not smiles_list:
        return 0.0
    
    # 只考虑有效的SMILES
    valid_smiles = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # 使用规范化的SMILES
                canonical = Chem.MolToSmiles(mol, canonical=True)
                valid_smiles.append(canonical)
        except:
            pass
    
    if not valid_smiles:
        return 0.0
    
    unique_smiles = set(valid_smiles)
    return len(unique_smiles) / len(valid_smiles)


def calculate_novelty(generated: List[str], reference: List[str]) -> float:
    """计算新颖性（生成的分子中不在参考集中的比例）"""
    if not generated:
        return 0.0
    
    # 规范化参考集
    reference_canonical = set()
    for smiles in reference:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                reference_canonical.add(canonical)
        except:
            pass
    
    # 计算新颖分子
    novel_count = 0
    valid_count = 0
    
    for smiles in generated:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                valid_count += 1
                if canonical not in reference_canonical:
                    novel_count += 1
        except:
            pass
    
    if valid_count == 0:
        return 0.0
    
    return novel_count / valid_count


def calculate_bleu_score(generated: List[str], target: List[str]) -> float:
    """计算BLEU分数（字符级别）"""
    if not generated or not target:
        return 0.0
    
    scores = []
    for gen, tgt in zip(generated, target):
        # 字符级别的BLEU
        gen_chars = list(gen)
        tgt_chars = list(tgt)
        
        # 计算n-gram overlap
        if len(gen_chars) == 0 or len(tgt_chars) == 0:
            scores.append(0.0)
            continue
            
        # 简单的字符重叠率作为BLEU近似
        common = set(gen_chars) & set(tgt_chars)
        score = len(common) / max(len(set(gen_chars)), len(set(tgt_chars)))
        scores.append(score)
    
    return np.mean(scores) if scores else 0.0


def calculate_exact_match(generated: List[str], target: List[str]) -> float:
    """计算精确匹配率"""
    if not generated or not target:
        return 0.0
    
    matches = 0
    for gen, tgt in zip(generated, target):
        try:
            # 使用规范化的SMILES比较
            mol_gen = Chem.MolFromSmiles(gen)
            mol_tgt = Chem.MolFromSmiles(tgt)
            
            if mol_gen is not None and mol_tgt is not None:
                canonical_gen = Chem.MolToSmiles(mol_gen, canonical=True)
                canonical_tgt = Chem.MolToSmiles(mol_tgt, canonical=True)
                
                if canonical_gen == canonical_tgt:
                    matches += 1
        except:
            pass
    
    return matches / len(generated)


def calculate_levenshtein_distance(generated: List[str], target: List[str]) -> float:
    """计算平均Levenshtein距离（归一化）"""
    if not generated or not target:
        return 1.0  # 最大距离
    
    distances = []
    for gen, tgt in zip(generated, target):
        # 计算编辑距离
        dist = Levenshtein.distance(gen, tgt)
        # 归一化
        max_len = max(len(gen), len(tgt))
        if max_len > 0:
            normalized_dist = dist / max_len
            distances.append(normalized_dist)
    
    return np.mean(distances) if distances else 1.0


def calculate_fingerprint_similarity(generated: List[str], target: List[str], 
                                    fingerprint_type: str = 'morgan') -> float:
    """计算分子指纹相似度"""
    if not generated or not target:
        return 0.0
    
    similarities = []
    
    for gen, tgt in zip(generated, target):
        try:
            mol_gen = Chem.MolFromSmiles(gen)
            mol_tgt = Chem.MolFromSmiles(tgt)
            
            if mol_gen is None or mol_tgt is None:
                continue
            
            if fingerprint_type == 'morgan':
                fp_gen = AllChem.GetMorganFingerprintAsBitVect(mol_gen, 2, 2048)
                fp_tgt = AllChem.GetMorganFingerprintAsBitVect(mol_tgt, 2, 2048)
            elif fingerprint_type == 'maccs':
                fp_gen = AllChem.GetMACCSKeysFingerprint(mol_gen)
                fp_tgt = AllChem.GetMACCSKeysFingerprint(mol_tgt)
            elif fingerprint_type == 'rdkit':
                fp_gen = Chem.RDKFingerprint(mol_gen)
                fp_tgt = Chem.RDKFingerprint(mol_tgt)
            else:
                continue
            
            # 计算Tanimoto相似度
            similarity = DataStructs.TanimotoSimilarity(fp_gen, fp_tgt)
            similarities.append(similarity)
            
        except:
            pass
    
    return np.mean(similarities) if similarities else 0.0


def calculate_fcd(generated: List[str], target: List[str]) -> float:
    """
    计算FCD分数的简化版本
    使用分子描述符的分布差异作为近似
    """
    if not generated or not target:
        return float('inf')
    
    def get_descriptors(smiles_list):
        """获取分子描述符"""
        descriptors = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    desc = [
                        Descriptors.MolWt(mol),
                        Descriptors.MolLogP(mol),
                        Descriptors.NumHDonors(mol),
                        Descriptors.NumHAcceptors(mol),
                        Descriptors.TPSA(mol),
                        Descriptors.NumRotatableBonds(mol),
                        Descriptors.NumAromaticRings(mol),
                        Descriptors.NumSaturatedRings(mol),
                    ]
                    descriptors.append(desc)
            except:
                pass
        return np.array(descriptors) if descriptors else np.array([])
    
    # 获取描述符
    gen_desc = get_descriptors(generated)
    tgt_desc = get_descriptors(target)
    
    if len(gen_desc) == 0 or len(tgt_desc) == 0:
        return float('inf')
    
    # 计算均值和协方差
    mu_gen = np.mean(gen_desc, axis=0)
    mu_tgt = np.mean(tgt_desc, axis=0)
    
    if len(gen_desc) > 1 and len(tgt_desc) > 1:
        cov_gen = np.cov(gen_desc.T)
        cov_tgt = np.cov(tgt_desc.T)
        
        # 简化的Frechet距离计算
        diff = mu_gen - mu_tgt
        # 避免奇异矩阵
        cov_mean = (cov_gen + cov_tgt) / 2
        cov_mean = cov_mean + np.eye(len(mu_gen)) * 1e-6
        
        try:
            # Frechet距离
            fcd = np.sqrt(np.sum(diff**2) + np.trace(cov_gen + cov_tgt - 2*cov_mean))
        except:
            # 如果计算失败，使用简单的均值距离
            fcd = np.sqrt(np.sum(diff**2))
    else:
        # 样本太少，只计算均值距离
        diff = mu_gen - mu_tgt
        fcd = np.sqrt(np.sum(diff**2))
    
    return float(fcd)


def calculate_all_metrics(generated: List[str], target: List[str]) -> Dict[str, float]:
    """计算所有评估指标"""
    metrics = {}
    
    # 1. 有效性
    metrics['validity'] = calculate_validity(generated)
    
    # 2. 唯一性
    metrics['uniqueness'] = calculate_uniqueness(generated)
    
    # 3. 新颖性
    metrics['novelty'] = calculate_novelty(generated, target)
    
    # 4. BLEU分数
    metrics['bleu_score'] = calculate_bleu_score(generated, target)
    
    # 5. 精确匹配
    metrics['exact_match'] = calculate_exact_match(generated, target)
    
    # 6. 编辑距离
    metrics['levenshtein_dist'] = calculate_levenshtein_distance(generated, target)
    
    # 7-9. 指纹相似度
    metrics['maccs_similarity'] = calculate_fingerprint_similarity(
        generated, target, fingerprint_type='maccs'
    )
    metrics['morgan_similarity'] = calculate_fingerprint_similarity(
        generated, target, fingerprint_type='morgan'
    )
    metrics['rdkit_similarity'] = calculate_fingerprint_similarity(
        generated, target, fingerprint_type='rdkit'
    )
    
    # 10. FCD分数
    metrics['fcd_score'] = calculate_fcd(generated, target)
    
    return metrics
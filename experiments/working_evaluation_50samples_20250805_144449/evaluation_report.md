# 分子生成模型评估报告（修复版）

**实验名称**: working_evaluation_50samples
**生成时间**: 2025-08-05 14:46:16
**测试样本数**: 50

## 评估结果

### 所有9个指标

| 序号 | 指标名称 | 分数 | 说明 |
|-----|---------|------|------|
| 1 | Validity | 0.7778 | 有效性 |
| 2 | Uniqueness | 1.0000 | 唯一性 |
| 3 | Novelty | 0.7500 | 新颖性 |
| 4 | BLEU Score | 0.6753 | BLEU分数 |
| 5 | Exact Match | 0.1944 | 完全匹配 |
| 6 | Levenshtein Distance | 0.6635 | 编辑距离相似度 |
| 7 | MACCS Similarity | 0.8551 | MACCS指纹相似度 |
| 8 | Morgan Similarity | 0.5692 | Morgan指纹相似度 |
| 9 | RDK Similarity | 0.7010 | RDK指纹相似度 |

**成功率**: 77.78% (28/36)

## 生成示例

### 示例 1
- **骨架**: `C=C1CC[C@@H]2C1CC(=O)[C@@H]1C3CCC(=O)C=C3CC[C@@H]21`
- **描述**: The molecule is a steroid ester that is methyl (17E)-pregna-4,17-dien-21-oate substituted by oxo gro...
- **目标**: `COC(=O)/C=C1/CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3C(=O)C[C@]12C`
- **生成**: `CC(=O)[C@H]1CC[C@H]2C3=C(CC[C@]12C)[C@@]1(C)CCC(=O)C(C)(C)[C@@H]1CC3`
- **有效**: ✓

### 示例 2
- **骨架**: `C1CC[C@H](OC[C@H]2OCC[C@@H](O[C@H]3CCCCO3)[C@@H]2O[C@H]2CCCCO2)OC1`
- **描述**: The molecule is a branched amino tetrasaccharide consisting of N-acetyl-beta-D-glucosamine having tw...
- **目标**: `CC(O)=N[C@@H]1[C@@H](O[C@@H]2O[C@@H](C)[C@@H](O)[C@@H](O)[C@@H]2O)[C@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2N=C(C)O)[C@@H](CO[C@@H]2O[C@@H](C)[C@@H](O)[C@@H](O)[C@@H]2O)O[C@H]1O`
- **生成**: `C[C@H]1[C@H]([C@H]([C@@H]([C@@H](O1)OC[C@@H]2[C@H]([C@@H]([C@H]([C@@H](O2)O)NC(=O)C)O[C@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CO)O[C@H]4[C@`
- **有效**: ✗

### 示例 3
- **骨架**: `N=C(Cc1c[nH]c2ccccc12)S[C@H]1CCCCO1`
- **描述**: The molecule is an indolylmethylglucosinolate that is the conjugate base of 4-methoxyglucobrassicin,...
- **目标**: `COc1cccc2[nH]cc(C/C(=N/OS(=O)(=O)[O-])S[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)c12`
- **生成**: `COc1cccc2[nH]cc(C/C(=N/OS(=O)(=O)[O-])S[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)c12`
- **有效**: ✓

### 示例 4
- **骨架**: `c1ccc(C2CCN(CCC(c3ccccc3)c3ccccc3)CC2)cc1`
- **描述**: The molecule is a synthetic piperidine derivative, effective against diarrhoea resulting from gastro...
- **目标**: `CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1`
- **生成**: `O=C(c1ccc(Cl)cc1)N1CCCCC1`
- **有效**: ✓

### 示例 5
- **骨架**: `N=c1[nH]cncc1C[n+]1ccsc1`
- **描述**: The molecule is a 1,3-thiazolium cation that is 1,3-thiazol-3-ium substituted by a methyl group at p...
- **目标**: `Cc1ncc(C[n+]2c(C(O)CCC(=O)O)sc(CCOP(=O)(O)OP(=O)(O)O)c2C)c(=N)[nH]1`
- **生成**: `Cc1ncc(C[n+]2csc(C(OCOP(=O)(O)O)C(C)C)c2C)c(=O)[nH]1`
- **有效**: ✓


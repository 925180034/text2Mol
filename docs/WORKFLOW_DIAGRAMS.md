# Text2Mol 工作流程图文档

## 🔄 系统工作流程总览

本文档详细描述了Text2Mol系统的各个工作流程，包括数据预处理、模型训练、推理生成和评估验证。

## 📊 1. 数据预处理流程

```mermaid
flowchart TD
    A[原始ChEBI-20数据集<br/>CSV格式] --> B{数据验证}
    B -->|有效| C[SMILES标准化<br/>RDKit处理]
    B -->|无效| X[丢弃无效记录]
    
    C --> D[多模态数据生成]
    
    D --> E[SMILES处理<br/>Tokenization]
    D --> F[Graph生成<br/>分子图构建]
    D --> G[Image生成<br/>2D结构图像]
    
    E --> H[SMILES数据<br/>train/val/test.csv]
    F --> I[Graph数据<br/>*.pkl格式]
    G --> J[Image数据<br/>*.pkl格式]
    
    subgraph "Graph特征提取"
        F1[原子特征<br/>类型、电荷、杂化]
        F2[化学键特征<br/>键型、芳香性]
        F3[PyTorch Geometric<br/>Data对象]
        F1 --> F3
        F2 --> F3
        F3 --> I
    end
    
    subgraph "Image特征提取" 
        G1[RDKit分子绘制<br/>2D结构图]
        G2[图像标准化<br/>299×299 RGB]
        G3[数据增强<br/>可选]
        G1 --> G2
        G2 --> G3
        G3 --> J
    end
    
    H --> K[数据加载器<br/>MultimodalDataset]
    I --> K
    J --> K
    
    style D fill:#e1f5fe
    style K fill:#c8e6c9
```

## 🎯 2. 模型训练流程

```mermaid
flowchart TD
    A[训练启动<br/>train_multimodal.py] --> B[配置加载<br/>YAML配置文件]
    B --> C[数据集初始化<br/>MultimodalDataset]
    C --> D[模型初始化<br/>End2EndMolecularGenerator]
    
    D --> E[多模态编码器初始化]
    E --> E1[SMILES编码器<br/>MolT5-Large]
    E --> E2[Graph编码器<br/>5-layer GIN]  
    E --> E3[Image编码器<br/>Swin Transformer]
    E --> E4[Text编码器<br/>BERT]
    
    E1 --> F[模态融合层<br/>Cross-Attention + Gating]
    E2 --> F
    E3 --> F  
    E4 --> F
    
    F --> G[MolT5生成器<br/>Adapter + Generator]
    
    G --> H{训练循环开始}
    H --> I[批次数据加载<br/>Scaffold + Text + Target]
    
    I --> J[前向传播<br/>Forward Pass]
    J --> K[损失计算<br/>CrossEntropyLoss]
    K --> L[反向传播<br/>Backward Pass]
    L --> M[参数更新<br/>AdamW优化器]
    
    M --> N{验证检查}
    N -->|每N步| O[验证集评估<br/>BLEU/Validity/Similarity]
    N -->|继续训练| H
    
    O --> P{早停检查}
    P -->|性能提升| Q[保存最佳模型<br/>best_model.pt]
    P -->|无提升| R[训练结束]
    
    Q --> H
    R --> S[最终模型保存<br/>final_model.pt]
    
    subgraph "损失函数"
        K1[生成损失<br/>Token-level CE]
        K2[对比损失<br/>可选]
        K1 --> K
        K2 --> K
    end
    
    subgraph "学习率调度"
        M1[Warmup阶段<br/>线性增长]
        M2[Cosine Decay<br/>余弦衰减]
        M1 --> M
        M2 --> M
    end
    
    style H fill:#ffecb3
    style O fill:#e8f5e8
    style S fill:#c8e6c9
```

## 🔮 3. 模型推理流程

```mermaid
flowchart TD
    A[推理请求<br/>Scaffold + Text] --> B[输入验证<br/>格式检查]
    B --> C[数据预处理<br/>标准化处理]
    
    C --> D{输入模态识别}
    D -->|SMILES| E1[SMILES编码器<br/>MolT5特征提取]
    D -->|Graph| E2[Graph编码器<br/>GIN特征提取]
    D -->|Image| E3[Image编码器<br/>Swin特征提取]
    
    C --> E4[Text编码器<br/>BERT特征提取]
    
    E1 --> F[模态融合<br/>768-dim统一特征]
    E2 --> F
    E3 --> F
    E4 --> F
    
    F --> G[MolT5适配器<br/>768→1024维度转换]
    G --> H[序列扩展<br/>Transformer输入格式]
    H --> I[MolT5生成<br/>Beam Search解码]
    
    I --> J[SMILES输出<br/>Generated Sequence]
    
    J --> K{目标输出模态}
    K -->|SMILES| L1[直接输出<br/>SMILES字符串]
    K -->|Graph| L2[SMILES→Graph解码器<br/>PyTorch Geometric]
    K -->|Image| L3[SMILES→Image解码器<br/>299×299 RGB]
    
    L2 --> M2[分子图对象<br/>节点+边+特征]
    L3 --> M3[分子图像数组<br/>标准化像素值]
    
    L1 --> N[输出验证<br/>有效性检查]
    M2 --> N
    M3 --> N
    
    N --> O[结果返回<br/>Generated Molecule]
    
    subgraph "生成参数控制"
        I1[Beam Size: 3-5<br/>多候选生成]
        I2[Temperature: 0.8<br/>随机性控制]  
        I3[Max Length: 128<br/>序列长度限制]
        I1 --> I
        I2 --> I
        I3 --> I
    end
    
    subgraph "质量控制"
        N1[分子有效性<br/>RDKit验证]
        N2[化学合理性<br/>规则检查]
        N3[Scaffold保持<br/>结构相似性]
        N1 --> N
        N2 --> N  
        N3 --> N
    end
    
    style F fill:#e1f5fe
    style I fill:#fff3e0
    style O fill:#c8e6c9
```

## 📈 4. 模型评估流程

```mermaid
flowchart TD
    A[评估启动<br/>real_model_evaluation.py] --> B[模型加载<br/>训练好的权重]
    B --> C[测试数据加载<br/>test.csv + pkl文件]
    
    C --> D[九种模态组合<br/>3输入×3输出]
    
    D --> E{模态循环}
    E --> F[样本批次处理<br/>100样本/组合]
    
    F --> G[模型推理<br/>生成分子]
    G --> H[输出后处理<br/>格式转换]
    
    H --> I[指标计算<br/>10种评估指标]
    
    subgraph "评估指标组"
        I1[分子质量指标<br/>Validity, Uniqueness, Novelty]
        I2[序列相似性指标<br/>BLEU, Exact Match, Levenshtein]
        I3[化学相似性指标<br/>MACCS, Morgan, RDK Fingerprints]
        I4[分布距离指标<br/>FCD Score]
        
        I1 --> I
        I2 --> I
        I3 --> I
        I4 --> I
    end
    
    I --> J[结果汇总<br/>JSON + Markdown]
    J --> K{所有组合完成?}
    K -->|否| E
    K -->|是| L[综合评估报告<br/>performance_summary.md]
    
    L --> M[可视化结果<br/>分子结构图]
    M --> N[评估完成<br/>结果保存]
    
    subgraph "质量验证"
        G1[生成失败检测<br/>INVALID标记]
        G2[Fallback机制移除<br/>真实性验证]  
        G3[设备一致性检查<br/>CUDA/CPU兼容]
        G1 --> G
        G2 --> G
        G3 --> G
    end
    
    subgraph "结果分析"
        L1[性能对比<br/>各模态表现]
        L2[错误模式分析<br/>失败原因统计]
        L3[化学有效性分析<br/>分子质量评估]
        L1 --> L
        L2 --> L
        L3 --> L
    end
    
    style I fill:#e8f5e8
    style L fill:#fff3e0
    style N fill:#c8e6c9
```

## ⚙️ 5. 配置管理流程

```mermaid
flowchart TD
    A[配置请求<br/>训练/推理/评估] --> B[配置文件选择<br/>configs/*.yaml]
    
    B --> C{配置类型}
    C -->|训练| D1[训练配置<br/>default_config.yaml]
    C -->|推理| D2[推理配置<br/>inference_config.yaml]  
    C -->|评估| D3[评估配置<br/>evaluation_config.yaml]
    
    D1 --> E[参数解析<br/>YAML → Dict]
    D2 --> E
    D3 --> E
    
    E --> F[参数验证<br/>类型检查+范围验证]
    F --> G{验证通过?}
    G -->|否| H[错误报告<br/>参数异常]
    G -->|是| I[配置应用<br/>系统初始化]
    
    I --> J[运行时调优<br/>动态参数调整]
    J --> K[日志记录<br/>配置快照]
    K --> L[执行完成<br/>配置归档]
    
    subgraph "配置层次"
        F1[全局配置<br/>系统级参数]
        F2[模型配置<br/>架构参数] 
        F3[训练配置<br/>训练超参]
        F4[硬件配置<br/>GPU/CPU设置]
        F1 --> F
        F2 --> F
        F3 --> F
        F4 --> F
    end
    
    style E fill:#e3f2fd
    style I fill:#e8f5e8
    style L fill:#c8e6c9
```

## 🔧 6. 错误处理和恢复流程

```mermaid
flowchart TD
    A[系统异常<br/>Exception Caught] --> B{异常类型}
    
    B -->|内存不足| C1[内存清理<br/>减少batch_size]
    B -->|设备错误| C2[设备检查<br/>CUDA可用性]
    B -->|数据错误| C3[数据验证<br/>格式检查]
    B -->|模型错误| C4[模型检查<br/>权重完整性]
    
    C1 --> D1[自动恢复<br/>降级运行]
    C2 --> D2[设备切换<br/>CPU Fallback] 
    C3 --> D3[数据跳过<br/>继续处理]
    C4 --> D4[模型重载<br/>backup权重]
    
    D1 --> E[恢复验证<br/>功能测试]
    D2 --> E
    D3 --> E
    D4 --> E
    
    E --> F{恢复成功?}
    F -->|是| G[继续执行<br/>记录警告]
    F -->|否| H[优雅退出<br/>保存状态]
    
    G --> I[错误统计<br/>性能监控]
    H --> J[错误报告<br/>详细日志]
    
    subgraph "监控系统"
        M1[性能监控<br/>资源使用率]
        M2[错误追踪<br/>异常统计]
        M3[自动报警<br/>关键错误]
        M1 --> I
        M2 --> I
        M3 --> I
    end
    
    style E fill:#fff3e0
    style G fill:#c8e6c9
    style H fill:#ffcdd2
```

## 📊 流程性能指标

### 时间复杂度
- **数据预处理**: O(n) - 线性于数据量
- **模型推理**: O(1) - 固定时间（单样本）
- **批量生成**: O(batch_size) - 线性于批次大小

### 空间复杂度
- **模型内存**: ~8GB（batch_size=2）
- **数据缓存**: ~2GB（完整数据集）
- **临时存储**: ~500MB（处理过程）

### 并发能力
- **多GPU训练**: 支持数据并行
- **多进程推理**: CPU/GPU异构计算
- **批量处理**: 最大batch_size=16

---

*该流程文档基于实际代码实现，最后更新: 2025-08-06*
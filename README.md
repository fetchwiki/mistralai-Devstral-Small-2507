# mistralai/Devstral-Small-2507

HuggingFace model: mistralai/Devstral-Small-2507

This is a mirror of the [mistralai/Devstral-Small-2507](https://huggingface.co/mistralai/Devstral-Small-2507) repository from HuggingFace.

**Note**: This repository contains metadata and configuration files only. The actual model files are stored on HuggingFace due to their large size.

## Repository Overview

| Type | Count | Total Size |
|------|-------|------------|
| Model Files | 11 | 87.84GB |
| Config Files | 2 | 773B |
| Code Files | 0 | 0B |
| Documentation | 1 | 18.33KB |

## Model Information

### Architecture Details

- **Architecture**: MistralForCausalLM
- **Model Type**: mistral
- **Vocabulary Size**: 131,072
- **Hidden Size**: 5,120
- **Number of Layers**: 40
- **Attention Heads**: 32
- **Max Position Embeddings**: 131,072

## Model Files

This repository contains metadata about the model files. The actual model files are stored on HuggingFace.

| File | Size | Type | Details |
|------|------|------|----------|
| [consolidated.safetensors](https://huggingface.co/mistralai/Devstral-Small-2507/blob/main/consolidated.safetensors) | 43GB | safetensors | 363 tensors |
| [model-00001-of-00010.safetensors](https://huggingface.co/mistralai/Devstral-Small-2507/blob/main/model-00001-of-00010.safetensors) | 4GB | safetensors | 32 tensors |
| [model-00002-of-00010.safetensors](https://huggingface.co/mistralai/Devstral-Small-2507/blob/main/model-00002-of-00010.safetensors) | 4GB | safetensors | 37 tensors |
| [model-00003-of-00010.safetensors](https://huggingface.co/mistralai/Devstral-Small-2507/blob/main/model-00003-of-00010.safetensors) | 4GB | safetensors | 37 tensors |
| [model-00004-of-00010.safetensors](https://huggingface.co/mistralai/Devstral-Small-2507/blob/main/model-00004-of-00010.safetensors) | 4GB | safetensors | 43 tensors |
| [model-00005-of-00010.safetensors](https://huggingface.co/mistralai/Devstral-Small-2507/blob/main/model-00005-of-00010.safetensors) | 4GB | safetensors | 37 tensors |
| [model-00006-of-00010.safetensors](https://huggingface.co/mistralai/Devstral-Small-2507/blob/main/model-00006-of-00010.safetensors) | 4GB | safetensors | 37 tensors |
| [model-00007-of-00010.safetensors](https://huggingface.co/mistralai/Devstral-Small-2507/blob/main/model-00007-of-00010.safetensors) | 4GB | safetensors | 43 tensors |
| [model-00008-of-00010.safetensors](https://huggingface.co/mistralai/Devstral-Small-2507/blob/main/model-00008-of-00010.safetensors) | 4GB | safetensors | 37 tensors |
| [model-00009-of-00010.safetensors](https://huggingface.co/mistralai/Devstral-Small-2507/blob/main/model-00009-of-00010.safetensors) | 4GB | safetensors | 37 tensors |
| [model-00010-of-00010.safetensors](https://huggingface.co/mistralai/Devstral-Small-2507/blob/main/model-00010-of-00010.safetensors) | 3GB | safetensors | 23 tensors |

**Total model files**: 11

## Tensor Analysis

### consolidated.safetensors

**Tensor Overview:**
- Total tensors: 363
- Total parameters: 23,572,403,200
- Data types: BF16 (363)
- Top layers: layers (360), norm (1), output (1), tok_embeddings (1)

**Sample Tensors:**
```
    - layers.0.attention.wk.weight: shape=[1024,5120], dtype=BF16, params=5,242,880
    - layers.0.attention.wo.weight: shape=[5120,4096], dtype=BF16, params=20,971,520
    - layers.0.attention.wq.weight: shape=[4096,5120], dtype=BF16, params=20,971,520
    - layers.0.attention.wv.weight: shape=[1024,5120], dtype=BF16, params=5,242,880
    - layers.0.attention_norm.weight: shape=[5120], dtype=BF16, params=5,120
```

### model-00001-of-00010.safetensors

**Tensor Overview:**
- Total tensors: 32
- Total parameters: 2,390,784,000
- Data types: BF16 (32)
- Top layers: model (32)

**Sample Tensors:**
```
    - model.embed_tokens.weight: shape=[131072,5120], dtype=BF16, params=671,088,640
    - model.layers.0.input_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.0.mlp.down_proj.weight: shape=[5120,32768], dtype=BF16, params=167,772,160
    - model.layers.0.mlp.gate_proj.weight: shape=[32768,5120], dtype=BF16, params=167,772,160
    - model.layers.0.mlp.up_proj.weight: shape=[32768,5120], dtype=BF16, params=167,772,160
```

### model-00002-of-00010.safetensors

**Tensor Overview:**
- Total tensors: 37
- Total parameters: 2,390,794,240
- Data types: BF16 (37)
- Top layers: model (37)

**Sample Tensors:**
```
    - model.layers.3.input_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.3.mlp.down_proj.weight: shape=[5120,32768], dtype=BF16, params=167,772,160
    - model.layers.3.mlp.gate_proj.weight: shape=[32768,5120], dtype=BF16, params=167,772,160
    - model.layers.3.mlp.up_proj.weight: shape=[32768,5120], dtype=BF16, params=167,772,160
    - model.layers.3.post_attention_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
```

### model-00003-of-00010.safetensors

**Tensor Overview:**
- Total tensors: 37
- Total parameters: 2,390,794,240
- Data types: BF16 (37)
- Top layers: model (37)

**Sample Tensors:**
```
    - model.layers.10.input_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.10.mlp.down_proj.weight: shape=[5120,32768], dtype=BF16, params=167,772,160
    - model.layers.10.mlp.gate_proj.weight: shape=[32768,5120], dtype=BF16, params=167,772,160
    - model.layers.10.mlp.up_proj.weight: shape=[32768,5120], dtype=BF16, params=167,772,160
    - model.layers.10.post_attention_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
```

### model-00004-of-00010.safetensors

**Tensor Overview:**
- Total tensors: 43
- Total parameters: 2,443,233,280
- Data types: BF16 (43)
- Top layers: model (43)

**Sample Tensors:**
```
    - model.layers.11.input_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.11.mlp.down_proj.weight: shape=[5120,32768], dtype=BF16, params=167,772,160
    - model.layers.11.post_attention_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.12.input_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.12.mlp.down_proj.weight: shape=[5120,32768], dtype=BF16, params=167,772,160
```

### model-00005-of-00010.safetensors

**Tensor Overview:**
- Total tensors: 37
- Total parameters: 2,390,794,240
- Data types: BF16 (37)
- Top layers: model (37)

**Sample Tensors:**
```
    - model.layers.16.input_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.16.mlp.down_proj.weight: shape=[5120,32768], dtype=BF16, params=167,772,160
    - model.layers.16.mlp.gate_proj.weight: shape=[32768,5120], dtype=BF16, params=167,772,160
    - model.layers.16.mlp.up_proj.weight: shape=[32768,5120], dtype=BF16, params=167,772,160
    - model.layers.16.post_attention_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
```

### model-00006-of-00010.safetensors

**Tensor Overview:**
- Total tensors: 37
- Total parameters: 2,390,794,240
- Data types: BF16 (37)
- Top layers: model (37)

**Sample Tensors:**
```
    - model.layers.20.input_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.20.mlp.down_proj.weight: shape=[5120,32768], dtype=BF16, params=167,772,160
    - model.layers.20.mlp.up_proj.weight: shape=[32768,5120], dtype=BF16, params=167,772,160
    - model.layers.20.post_attention_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.21.input_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
```

### model-00007-of-00010.safetensors

**Tensor Overview:**
- Total tensors: 43
- Total parameters: 2,443,233,280
- Data types: BF16 (43)
- Top layers: model (43)

**Sample Tensors:**
```
    - model.layers.24.input_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.24.mlp.down_proj.weight: shape=[5120,32768], dtype=BF16, params=167,772,160
    - model.layers.24.post_attention_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.25.input_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.25.mlp.down_proj.weight: shape=[5120,32768], dtype=BF16, params=167,772,160
```

### model-00008-of-00010.safetensors

**Tensor Overview:**
- Total tensors: 37
- Total parameters: 2,390,794,240
- Data types: BF16 (37)
- Top layers: model (37)

**Sample Tensors:**
```
    - model.layers.29.input_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.29.mlp.down_proj.weight: shape=[5120,32768], dtype=BF16, params=167,772,160
    - model.layers.29.mlp.gate_proj.weight: shape=[32768,5120], dtype=BF16, params=167,772,160
    - model.layers.29.mlp.up_proj.weight: shape=[32768,5120], dtype=BF16, params=167,772,160
    - model.layers.29.post_attention_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
```

### model-00009-of-00010.safetensors

**Tensor Overview:**
- Total tensors: 37
- Total parameters: 2,390,794,240
- Data types: BF16 (37)
- Top layers: model (37)

**Sample Tensors:**
```
    - model.layers.33.input_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.33.mlp.down_proj.weight: shape=[5120,32768], dtype=BF16, params=167,772,160
    - model.layers.33.mlp.up_proj.weight: shape=[32768,5120], dtype=BF16, params=167,772,160
    - model.layers.33.post_attention_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.34.input_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
```

### model-00010-of-00010.safetensors

**Tensor Overview:**
- Total tensors: 23
- Total parameters: 1,950,387,200
- Data types: BF16 (23)
- Top layers: model (22), lm_head (1)

**Sample Tensors:**
```
    - lm_head.weight: shape=[131072,5120], dtype=BF16, params=671,088,640
    - model.layers.37.input_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.37.mlp.down_proj.weight: shape=[5120,32768], dtype=BF16, params=167,772,160
    - model.layers.37.post_attention_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
    - model.layers.38.input_layernorm.weight: shape=[5120], dtype=BF16, params=5,120
```

## Configuration Files

| File | Category | Size |
|------|----------|------|
| config.json | model | 641B |
| generation_config.json | generation | 132B |

## Usage

To use this model, visit the original HuggingFace repository:
- [mistralai/Devstral-Small-2507](https://huggingface.co/mistralai/Devstral-Small-2507)

## Additional Information

This mirror was created to provide easy access to model metadata and configuration files. For the actual model weights and full functionality, please visit the original repository on HuggingFace.

---

**Repository Statistics:**
- Total files analyzed: 36
- Total size: 87.84GB
- Model files: 11
- Generated: 2025-07-14

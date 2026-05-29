# DeepSeek V4 Pro Execution Path

本文总结 `DeepSeek-V4-Pro` 在公开模型结构和本地 `vllm + vllm-ascend` 代码中的执行链路。这里的 `v4pro` 按公开仓库和模型卡理解为 `deepseek-ai/DeepSeek-V4-Pro`。

## 1. 核心结论

`DeepSeek-V4-Pro` 是 DeepSeek-V4 系列里的 Pro 模型，公开模型卡说明它是 MoE 模型，总参数约 `1.6T`，每 token 激活约 `49B` 参数，支持 `1M` token context。技术报告和 `config.json` 进一步给出了模型结构：`61` 层 Transformer，`hidden_size = 7168`，`num_attention_heads = 128`，`head_dim = 512`，`num_key_value_heads = 1`，所有 Transformer block 都使用 MoE FFN。

注意：本地 `vllm` 已经有通用 `DeepseekV4ForCausalLM` 实现；本地 `vllm-ascend` 又注册了 Ascend 版本 `AscendDeepseekV4ForCausalLM`，所以在 NPU 上跑时，模型注册会优先落到 `vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM`。

公开结构参数：

| 项目 | DeepSeek-V4-Pro |
| --- | --- |
| architecture | `DeepseekV4ForCausalLM` |
| model_type | `deepseek_v4` |
| total params | `1.6T` |
| activated params | `49B` |
| context length | `1,048,576` / 1M |
| Transformer layers | `61` |
| hidden size | `7168` |
| attention heads | `128` |
| head dim | `512` |
| KV heads | `1` |
| q_lora_rank | `1536` |
| o_lora_rank | `1024` |
| qk_rope_head_dim | `64` |
| routed experts | `384` |
| shared experts | `1` |
| activated routed experts / token | `6` |
| MoE intermediate size | `3072` |
| hash MoE layers | first `3` MoE layers |
| MTP depth | `1` |
| expert dtype | `fp4` |
| common quantization | `fp8` |

## 2. 模型执行时序图

```mermaid
sequenceDiagram
    autonumber
    actor User as User
    participant API as vllm serve / LLM(...)
    participant Engine as vLLM EngineCore
    participant Scheduler as Scheduler
    participant Executor as Executor
    participant Platform as NPUPlatform
    participant Worker as NPUWorker
    participant Runner as NPUModelRunner
    participant Registry as ModelRegistry / get_model
    participant Model as AscendDeepseekV4ForCausalLM
    participant Layer as DeepseekV4DecoderLayer x 61
    participant Ops as Ascend attention / MoE ops
    participant NPU as torch-npu / CANN / NPU

    User->>API: vllm serve deepseek-ai/DeepSeek-V4-Pro<br/>or LLM(model=...)
    API->>Engine: process input / add request
    Engine->>Scheduler: schedule tokens, blocks, KV cache
    Scheduler-->>Engine: scheduler_output

    Engine->>Executor: execute_model(scheduler_output)
    Executor->>Platform: platform plugin selects NPUPlatform
    Platform-->>Executor: worker_cls = NPUWorker<br/>attention backend = Ascend
    Executor->>Worker: RPC execute_model(...)
    Worker->>Runner: execute_model(...)

    alt first load / initialization
        Worker->>Runner: load_model()
        Runner->>Registry: get_model(vllm_config)
        Registry->>Model: construct AscendDeepseekV4ForCausalLM
        Model-->>Runner: model ready
    end

    Runner->>Runner: update request states
    Runner->>Runner: prepare input_ids / positions / attention metadata
    Runner->>Runner: set_ascend_forward_context(...)
    Runner->>Model: forward(input_ids, positions, ...)

    loop 61 Transformer layers
        Model->>Layer: layer.forward(...)
        Layer->>Layer: mHC pre + RMSNorm
        Layer->>Ops: DeepseekV4Attention<br/>HCA or CSA
        Ops->>NPU: compressed KV / sparse MLA / SWA cache ops
        NPU-->>Ops: attention output
        Ops-->>Layer: hidden states
        Layer->>Layer: mHC post
        Layer->>Layer: mHC pre + RMSNorm
        Layer->>Ops: DeepseekV4MoE<br/>1 shared + 384 routed experts
        Ops->>NPU: gate + top6 experts + fused/mega MoE
        NPU-->>Ops: MoE output
        Layer->>Layer: mHC post
        Layer-->>Model: hidden states
    end

    Model-->>Runner: final hidden states
    Runner->>Model: compute_logits(sample_hidden_states)
    Model-->>Runner: logits
    Runner->>Runner: sample_tokens(...)
    Runner-->>Worker: ModelRunnerOutput
    Worker-->>Engine: token output
    Engine-->>API: RequestOutput
    API-->>User: streaming / final response
```

## 3. 模型结构概览图

```mermaid
flowchart LR
    Input["Input tokens<br/>DeepSeek-V4 tokenizer<br/>vocab_size = 129280"]

    subgraph Model["DeepSeek-V4-Pro<br/>DeepseekV4ForCausalLM"]
        Embed["Token embedding<br/>hidden_size = 7168"]
        Blocks["61 Transformer blocks"]
        Norm["Final RMSNorm<br/>rms_norm_eps = 1e-6"]
        Head["LM head<br/>vocab_size = 129280"]
    end

    subgraph Attention["Hybrid Attention"]
        HCA["HCA<br/>Heavily Compressed Attention<br/>compression ratio = 128"]
        CSA["CSA<br/>Compressed Sparse Attention<br/>compression ratio = 4<br/>index_topk = 1024"]
        SWA["Sliding Window branch<br/>window = 128"]
    end

    subgraph MoE["DeepSeekMoE FFN"]
        Shared["1 shared expert"]
        Routed["384 routed experts"]
        TopK["top-6 routed experts / token"]
        ExpertFFN["expert intermediate size = 3072"]
    end

    subgraph Other["Other model mechanisms"]
        MHC["mHC<br/>Manifold-Constrained Hyper-Connections<br/>hc_mult = 4"]
        MTP["MTP depth = 1"]
        Quant["FP4 experts + FP8 common weights"]
    end

    Input --> Embed --> Blocks --> Norm --> Head
    Blocks --> Attention
    Blocks --> MoE
    Blocks --> Other
    HCA --> SWA
    CSA --> SWA
    Shared --> ExpertFFN
    Routed --> TopK --> ExpertFFN
```

## 4. 模型结构图

```mermaid
flowchart TB
    Input["input_ids + positions"]
    Embedding["VocabParallelEmbedding<br/>129280 x 7168"]

    subgraph Stack["61 x DeepseekV4DecoderLayer"]
        Pattern["Layer attention pattern<br/>Layer 0-1: HCA<br/>Layer 2-60: CSA / HCA interleaved"]
        HashMoE["MoE routing pattern<br/>first 3 MoE layers: Hash routing<br/>remaining layers: noaux_tc routing"]
    end

    subgraph Decoder["One DeepseekV4DecoderLayer"]
        ResidualA["residual stream"]
        MHCPreA["mHC pre for attention<br/>mhc_pre"]
        AttnNorm["attn_norm<br/>RMSNorm"]
        Attn["DeepseekV4Attention"]
        MHCPostA["mHC post for attention<br/>mhc_post"]
        MHCPreF["mHC pre for FFN<br/>mhc_pre"]
        FFNNorm["ffn_norm<br/>RMSNorm"]
        FFN["DeepseekV4MoE"]
        MHCPostF["mHC post for FFN<br/>mhc_post"]
    end

    subgraph AttnDetail["DeepseekV4Attention detail"]
        QKV["fused_wqa_wkv<br/>q_lora_rank = 1536<br/>kv_lora_rank = head_dim = 512"]
        QNorm["q_norm + wq_b"]
        KVNorm["kv_norm"]
        RoPE["DeepSeek V4 RoPE<br/>qk_rope_head_dim = 64"]
        Indexer{"compress_ratio == 4 ?"}
        CSAPath["CSA path<br/>DeepseekV4Indexer<br/>index_heads = 64<br/>index_head_dim = 128<br/>index_topk = 1024"]
        HCAPath["HCA path<br/>dense attention over heavily compressed KV<br/>compress_ratio = 128"]
        SWAPath["SWA branch<br/>sliding_window = 128"]
        MLA["DeepseekV4MultiHeadLatentAttentionWrapper<br/>num_heads = 128<br/>head_dim = 512<br/>o_groups = 16"]
        OProj["wo_a + wo_b<br/>o_lora_rank = 1024"]
    end

    subgraph MoEDetail["DeepseekV4MoE detail"]
        Gate["GateLinear<br/>384 routed experts"]
        Route{"layer < 3 ?"}
        Hash["Hash routing<br/>tid2eid"]
        NoAux["noaux_tc routing<br/>sqrtsoftplus scoring<br/>norm_topk_prob = true"]
        Top6["select top-6 experts"]
        SharedExpert["shared expert x 1"]
        RoutedExpert["routed expert FFN<br/>intermediate = 3072<br/>SwiGLU clamp limit = 10"]
        Combine["combine shared + routed output<br/>routed_scaling_factor = 2.5"]
    end

    OutputNorm["Final RMSNorm"]
    LMHead["ParallelLMHead + LogitsProcessor"]

    Input --> Embedding --> Stack --> Decoder --> OutputNorm --> LMHead
    Pattern --> Decoder
    HashMoE --> Decoder

    ResidualA --> MHCPreA --> AttnNorm --> Attn --> MHCPostA --> MHCPreF --> FFNNorm --> FFN --> MHCPostF

    Attn --> QKV
    QKV --> QNorm
    QKV --> KVNorm
    QNorm --> RoPE
    KVNorm --> RoPE
    RoPE --> Indexer
    Indexer -->|yes, ratio 4| CSAPath
    Indexer -->|no, ratio 128| HCAPath
    CSAPath --> SWAPath
    HCAPath --> SWAPath
    SWAPath --> MLA --> OProj

    FFN --> Gate --> Route
    Route -->|first 3 MoE layers| Hash
    Route -->|other layers| NoAux
    Hash --> Top6
    NoAux --> Top6
    Top6 --> RoutedExpert
    SharedExpert --> Combine
    RoutedExpert --> Combine
```

## 5. 本地代码对应关系

执行入口和框架侧：

- `vllm serve` CLI：`vllm/entrypoints/cli/serve.py`
- 离线 `LLM(...)`：`vllm/entrypoints/llm.py`
- vLLM engine：`vllm/v1/engine/llm_engine.py`、`vllm/v1/engine/async_llm.py`
- 调度和执行：`vllm/v1/engine/core.py`
- Ascend platform：`vllm_ascend/platform.py`
- Ascend worker：`vllm_ascend/worker/worker.py`
- Ascend model runner：`vllm_ascend/worker/model_runner_v1.py`

模型注册和模型侧：

- vLLM 原生注册：`vllm/model_executor/models/registry.py`
- vLLM 原生实现：`vllm/model_executor/models/deepseek_v4.py`
- vllm-ascend 覆盖注册：`vllm_ascend/models/__init__.py`
- vllm-ascend Ascend 实现：`vllm_ascend/models/deepseek_v4.py`
- DeepSeek-V4 attention kernel/interface：`vllm/model_executor/layers/deepseek_v4_attention.py`
- Ascend DSA / compressed attention：`vllm_ascend/attention/dsa_v1.py`、`vllm_ascend/ops/dsa.py`
- DeepSeek-V4 compressor patch：`vllm_ascend/patch/worker/patch_deepseek_compressor.py`

关键代码点：

- `vllm_ascend.models.__init__.py` 注册 `DeepseekV4ForCausalLM -> AscendDeepseekV4ForCausalLM`。
- `AscendDeepseekV4ForCausalLM` 内部创建 `DeepseekV4Model`、`lm_head` 和 `LogitsProcessor`。
- `DeepseekV4Model` 根据 `config.num_hidden_layers` 创建 61 层 `DeepseekV4DecoderLayer`。
- 每个 `DeepseekV4DecoderLayer` 包含 `DeepseekV4Attention`、`DeepseekV4MoE`、两组 RMSNorm 和两段 mHC pre/post。
- `DeepseekV4Attention` 根据 `config.compress_ratios[layer_id]` 区分 HCA / CSA：`compress_ratio = 128` 表示 HCA，`compress_ratio = 4` 表示 CSA，CSA 层额外创建 `DeepseekV4Indexer`。
- `DeepseekV4MoE` 读取 `n_routed_experts = 384`、`num_experts_per_tok = 6`、`n_shared_experts = 1`、`moe_intermediate_size = 3072`；前 3 层使用 Hash routing，后续使用 `noaux_tc`。

## 6. 来源

- Hugging Face model card: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro>
- Hugging Face `config.json`: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json>
- DeepSeek-V4 technical report PDF: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf>


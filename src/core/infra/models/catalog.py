"""Static provider catalog for ``GET /models/providers``.

Mirrors ``internal/types/model.go::GetModelTypes`` on the Go side. Each
entry exposes the ``ProviderTypeMeta`` contract fields the UI uses to
render the create-model form: a value identifier, a human-readable
label, a one-line description, the default per-model-type endpoint
URLs, and the set of model types the provider supports.

The model types in this catalog use the **frontend alias** form
(``chat`` / ``embedding`` / ``rerank`` / ``vllm`` / ``asr``) — the
backend enum form (``KnowledgeQA`` / ``Embedding`` / ``Rerank`` /
``VLLM`` / ``ASR``) is what ``POST /models`` accepts in the request
body. See ``docs/api/model.md`` for the mapping.
"""

from __future__ import annotations

from src.core.contracts.infra import ProviderTypeMeta

# Type aliases (frontend-facing, as documented in docs/api/model.md).
_CHAT = "chat"
_EMBEDDING = "embedding"
_RERANK = "rerank"
_VLLM = "vllm"
_ASR = "asr"


PROVIDER_CATALOG: tuple[ProviderTypeMeta, ...] = (
    ProviderTypeMeta(
        value="generic",
        label="自定义 (OpenAI 兼容接口)",
        description="Custom OpenAI-compatible endpoint",
        defaultUrls={
            _CHAT: "",
            _EMBEDDING: "",
            _RERANK: "",
        },
        modelTypes=[_CHAT, _EMBEDDING, _RERANK, _VLLM],
    ),
    ProviderTypeMeta(
        value="openai",
        label="OpenAI",
        description="GPT, text-embedding-3, etc.",
        defaultUrls={
            _CHAT: "https://api.openai.com/v1",
            _EMBEDDING: "https://api.openai.com/v1",
            _RERANK: "",
        },
        modelTypes=[_CHAT, _EMBEDDING, _RERANK, _VLLM],
    ),
    ProviderTypeMeta(
        value="aliyun",
        label="阿里云 DashScope",
        description="qwen-plus, tongyi-embedding-vision-plus, qwen3-rerank, etc.",
        defaultUrls={
            _CHAT: "https://dashscope.aliyuncs.com/compatible-mode/v1",
            _EMBEDDING: "https://dashscope.aliyuncs.com/compatible-mode/v1",
            _RERANK: "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        },
        modelTypes=[_CHAT, _EMBEDDING, _RERANK, _VLLM],
    ),
    ProviderTypeMeta(
        value="zhipu",
        label="智谱 BigModel",
        description="glm-4.7, embedding-3, rerank, etc.",
        defaultUrls={
            _CHAT: "https://open.bigmodel.cn/api/paas/v4",
            _EMBEDDING: "https://open.bigmodel.cn/api/paas/v4/embeddings",
            _RERANK: "https://open.bigmodel.cn/api/paas/v4/rerank",
        },
        modelTypes=[_CHAT, _EMBEDDING, _RERANK, _VLLM],
    ),
    ProviderTypeMeta(
        value="volcengine",
        label="火山引擎 Volcengine",
        description="doubao-pro, VikingDB rerank, etc.",
        defaultUrls={
            _CHAT: "https://ark.cn-beijing.volces.com/api/v3",
            _EMBEDDING: "https://ark.cn-beijing.volces.com/api/v3",
            _RERANK: "https://api-knowledgebase.mlp.cn-beijing.volces.com",
        },
        modelTypes=[_CHAT, _EMBEDDING, _RERANK, _VLLM],
    ),
    ProviderTypeMeta(
        value="hunyuan",
        label="腾讯混元 Hunyuan",
        description="hunyuan-pro, hunyuan-embedding, etc.",
        defaultUrls={
            _CHAT: "https://api.hunyuan.tencent.com/v1",
            _EMBEDDING: "https://api.hunyuan.tencent.com/v1",
        },
        modelTypes=[_CHAT, _EMBEDDING],
    ),
    ProviderTypeMeta(
        value="deepseek",
        label="DeepSeek",
        description="deepseek-chat, deepseek-reasoner",
        defaultUrls={
            _CHAT: "https://api.deepseek.com/v1",
        },
        modelTypes=[_CHAT],
    ),
    ProviderTypeMeta(
        value="minimax",
        label="MiniMax",
        description="abab series",
        defaultUrls={
            _CHAT: "https://api.minimax.chat/v1",
        },
        modelTypes=[_CHAT],
    ),
    ProviderTypeMeta(
        value="mimo",
        label="小米 MiMo",
        description="Xiaomi MiMo chat models",
        defaultUrls={
            _CHAT: "https://api.xiaomimimo.com/v1",
        },
        modelTypes=[_CHAT],
    ),
    ProviderTypeMeta(
        value="siliconflow",
        label="硅基流动 SiliconFlow",
        description="Qwen / DeepSeek / BGE on SiliconFlow",
        defaultUrls={
            _CHAT: "https://api.siliconflow.cn/v1",
            _EMBEDDING: "https://api.siliconflow.cn/v1",
            _RERANK: "https://api.siliconflow.cn/v1/rerank",
        },
        modelTypes=[_CHAT, _EMBEDDING, _RERANK, _VLLM],
    ),
    ProviderTypeMeta(
        value="jina",
        label="Jina",
        description="jina-embeddings-v3, jina-reranker-v2",
        defaultUrls={
            _EMBEDDING: "https://api.jina.ai/v1",
            _RERANK: "https://api.jina.ai/v1",
        },
        modelTypes=[_EMBEDDING, _RERANK],
    ),
    ProviderTypeMeta(
        value="openrouter",
        label="OpenRouter",
        description="Aggregated multi-provider routing",
        defaultUrls={
            _CHAT: "https://openrouter.ai/api/v1",
        },
        modelTypes=[_CHAT, _VLLM],
    ),
    ProviderTypeMeta(
        value="requesty",
        label="Requesty",
        description="Aggregated multi-provider routing",
        defaultUrls={
            _CHAT: "https://router.requesty.ai/v1",
            _EMBEDDING: "https://router.requesty.ai/v1",
        },
        modelTypes=[_CHAT, _EMBEDDING, _VLLM],
    ),
    ProviderTypeMeta(
        value="gemini",
        label="Google Gemini",
        description="gemini-2.0-flash, gemini-1.5-pro",
        defaultUrls={
            _CHAT: "https://generativelanguage.googleapis.com/v1beta",
        },
        modelTypes=[_CHAT],
    ),
    ProviderTypeMeta(
        value="modelscope",
        label="魔搭 ModelScope",
        description="Qwen / BGE on ModelScope",
        defaultUrls={
            _CHAT: "https://api-inference.modelscope.cn/v1",
            _EMBEDDING: "https://api-inference.modelscope.cn/v1",
        },
        modelTypes=[_CHAT, _EMBEDDING, _VLLM],
    ),
    ProviderTypeMeta(
        value="moonshot",
        label="月之暗面 Moonshot",
        description="moonshot-v1-8k, moonshot-v1-128k",
        defaultUrls={
            _CHAT: "https://api.moonshot.cn/v1",
        },
        modelTypes=[_CHAT, _VLLM],
    ),
    ProviderTypeMeta(
        value="qianfan",
        label="百度千帆 Baidu Cloud",
        description="ERNIE-Bot, BGE on Qianfan",
        defaultUrls={
            _CHAT: "https://qianfan.baidubce.com/v2",
            _EMBEDDING: "https://qianfan.baidubce.com/v2",
            _RERANK: "https://qianfan.baidubce.com/v2",
        },
        modelTypes=[_CHAT, _EMBEDDING, _RERANK, _VLLM],
    ),
    ProviderTypeMeta(
        value="qiniu",
        label="七牛云 Qiniu",
        description="Qiniu LLM gateway",
        defaultUrls={
            _CHAT: "https://api.qnaigc.com/v1",
        },
        modelTypes=[_CHAT],
    ),
    ProviderTypeMeta(
        value="longcat",
        label="LongCat AI",
        description="LongCat chat models",
        defaultUrls={
            _CHAT: "https://api.longcat.chat/v1",
        },
        modelTypes=[_CHAT],
    ),
    ProviderTypeMeta(
        value="gpustack",
        label="GPUStack",
        description="Self-hosted GPUStack inference",
        defaultUrls={
            _CHAT: "",
            _EMBEDDING: "",
            _RERANK: "",
        },
        modelTypes=[_CHAT, _EMBEDDING, _RERANK, _VLLM],
    ),
    ProviderTypeMeta(
        value="xunfei",
        label="科大讯飞 (ASR)",
        description="iFlyTek Spark ASR models",
        defaultUrls={
            _ASR: "wss://iat-api.xfyun.cn/v2/iat",
        },
        modelTypes=[_ASR],
    ),
)


def filter_providers(
    catalog: tuple[ProviderTypeMeta, ...],
    *,
    model_type: str | None = None,
) -> list[ProviderTypeMeta]:
    """Return catalog entries supporting ``model_type`` (frontend alias).

    An empty / ``None`` model_type returns the full catalog. Comparison
    is membership-based so providers supporting multiple types (e.g.
    ``aliyun`` supports chat/embedding/rerank/vllm) appear in each
    filter view.
    """
    if not model_type:
        return list(catalog)
    return [p for p in catalog if p.model_types and model_type in p.model_types]


__all__ = ["PROVIDER_CATALOG", "filter_providers"]

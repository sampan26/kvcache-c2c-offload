"""
This script demonstrates the KV cache reuse problem when offloading is disabled.

**The Problem:**
When GPU KV cache is limited and multiple requests with recurring prompts arrive,
the cache manager must evict older cache entries. Without offloading to host memory,
these evicted entries are permanently lost and must be recomputed from scratch when
the same prompt appears again.

**This Demo Shows:**
- A constrained GPU cache that can only hold one request at a time
- Alternating between two different prompts (simulating multi-tenant workload)
- Zero cache reuse because evicted blocks are discarded, not preserved
- Every repeated prompt pays full recomputation cost

** Usage: **
- TLLM_LOG_LEVEL=DEBUG python intro.py 2>&1 | grep -i reuse

**Expected Output:**
- Debug logs showing "reused blocks: 0" 
- Cache hit rate: 0%
- Each prompt A repetition recomputes from scratch despite being identical
"""



from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig


def main():
    # Three prompts that will compete for limited cache space
    # Each is long enough to fill significant cache capacity
    prompts = [
        "Explain quantum computing. " + "Provide detailed examples and use cases. " * 50,  # ~600 tokens
        "Describe machine learning. " + "Include algorithms and applications. " * 50,      # ~600 tokens  
    ]

    kv_cache_max_tokens = 300
    
    print(f"GPU cache capacity: {kv_cache_max_tokens} tokens")
    print(f"Cache fits: 1 prompts max")
    print(f"Host offloading: DISABLED\n")

    llm = LLM(
        model="Qwen/Qwen3-8B",
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=True,
            max_tokens=kv_cache_max_tokens,
            host_cache_size=0  # No offloading
        )
    )

    sampling_params = SamplingParams(max_tokens=32)

    # Do multiple back-and-forth passes to exhaust cache
    for _ in range(3):
        llm.generate(prompts[0], sampling_params)
        llm.generate(prompts[1], sampling_params)

    llm.shutdown()


if __name__ == "__main__":
    main()
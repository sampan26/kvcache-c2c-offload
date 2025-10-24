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

**Expected Output:**
- Debug logs showing "reused blocks: 0" 
- Cache hit rate: 0%
- Each prompt A repetition recomputes from scratch despite being identical
"""

import argparse
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig


def main():
    print("=" * 80)
    print("KV CACHE REUSE PROBLEM DEMONSTRATION")
    print("Scenario: GPU cache too small + No offloading = Zero reuse")
    print("=" * 80)
    print()

    # Two distinct prompts simulating different requests
    # In production, think: different system prompts or different RAG contexts
    prompt_a = (
        "You are a helpful AI assistant specialized in Python programming. "
        "Please explain how to implement a binary search algorithm."
    )
    
    prompt_b = (
        "You are a creative writing assistant focused on storytelling. "
        "Please write the opening paragraph of a mystery novel."
    )

    # Configuration that exposes the problem
    max_batch_size = 1  # Sequential processing
    max_seq_len = 512
    
    # CRITICAL: Small GPU cache (only 512 tokens worth)
    # This means only ONE request's KV cache can fit at a time
    kv_cache_max_tokens = 512
    kv_cache_page_size = 16
    
    # NO HOST CACHE - This is the problem we're demonstrating
    kv_cache_host_size = 0
    
    print("Configuration:")
    print(f"  - GPU KV cache capacity: {kv_cache_max_tokens} tokens")
    print(f"  - Host cache size: {kv_cache_host_size} bytes (DISABLED)")
    print(f"  - Cache block size: {kv_cache_page_size} tokens")
    print(f"  - Batch size: {max_batch_size} (sequential requests)")
    print()
    print("What this means: When prompt B arrives, prompt A's cache MUST be evicted.")
    print("With no host cache, that evicted data is LOST forever.")
    print()

    sampling_params = SamplingParams(max_tokens=64, temperature=0.7)

    # Initialize LLM with constrained cache and NO offloading
    llm = LLM(
        model="Qwen/Qwen3-8B",
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=True,  # Reuse is ENABLED, but...
            max_tokens=kv_cache_max_tokens,
            tokens_per_block=kv_cache_page_size,
            host_cache_size=kv_cache_host_size  # ...nowhere to offload to!
        )
    )

    print("-" * 80)
    print("ROUND 1: Initial requests (cold cache)")
    print("-" * 80)
    
    # Request 1: Prompt A (first time)
    print("\n[Request 1] Processing Prompt A (first occurrence)")
    print("Expected: Full computation, cache stored on GPU")
    output_a1 = llm.generate(prompt_a, sampling_params)
    print(f"✓ Generated: {output_a1.outputs[0].text[:80]}...")
    print("   Cache status: Prompt A is now cached on GPU")
    
    # Request 2: Prompt B (first time)
    print("\n[Request 2] Processing Prompt B (first occurrence)")
    print("Expected: Full computation, EVICTS Prompt A from GPU")
    output_b1 = llm.generate(prompt_b, sampling_params)
    print(f"✓ Generated: {output_b1.outputs[0].text[:80]}...")
    print("   Cache status: Prompt B cached, Prompt A EVICTED (lost forever)")

    print("\n" + "=" * 80)
    print("ROUND 2: Repeated requests (should see cache reuse... but won't!)")
    print("=" * 80)
    
    # Request 3: Prompt A again
    print("\n[Request 3] Processing Prompt A (SECOND TIME - should reuse cache)")
    print("Problem: Prompt A's cache was evicted in Request 2")
    print("Expected: 'reused blocks: 0' - must recompute from scratch!")
    output_a2 = llm.generate(prompt_a, sampling_params)
    print(f"✓ Generated: {output_a2.outputs[0].text[:80]}...")
    print("   ⚠️  NO CACHE REUSE - Full recomputation despite identical prompt")
    
    # Request 4: Prompt B again
    print("\n[Request 4] Processing Prompt B (SECOND TIME - should reuse cache)")
    print("Problem: Prompt B's cache was just evicted in Request 3")
    print("Expected: 'reused blocks: 0' - must recompute from scratch!")
    output_b2 = llm.generate(prompt_b, sampling_params)
    print(f"✓ Generated: {output_b2.outputs[0].text[:80]}...")
    print("   ⚠️  NO CACHE REUSE - Full recomputation despite identical prompt")

    print("\n" + "=" * 80)
    print("RESULTS: The Cache Thrashing Problem")
    print("=" * 80)
    print("""
Summary:
  - 4 requests processed, but only 2 unique prompts
  - Without offloading: Each repeated prompt must recompute from scratch
  - Cache hit rate: 0%
  - Reused blocks: 0
  
The Waste:
  - 2 full redundant computations that could have been avoided
  - GPU cycles wasted on identical work
  - Increased latency for users
  - Lower throughput (fewer requests/second)

In production with hundreds of requests sharing common prefixes (system prompts,
RAG contexts), this cache thrashing becomes devastating to performance.

SOLUTION: Enable host cache offloading to preserve evicted KV cache entries...
(See next demo with --enable_offloading flag)
""")

    llm.shutdown()
    
    print("\n" + "=" * 80)
    print("To see this in DEBUG logs, run:")
    print("  TLLM_LOG_LEVEL=DEBUG python kv_cache_problem_demo.py 2>&1 | grep -i reuse")
    print("\nLook for lines showing:")
    print("  'reused blocks: 0' <- The smoking gun of cache thrashing")
    print("=" * 80)


if __name__ == "__main__":
    main()

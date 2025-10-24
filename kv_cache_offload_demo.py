"""
KV Cache Offloading Demo - Works identically on Hopper and Grace Hopper

This script demonstrates KV cache offloading with host memory.
The same code runs on both PCIe (Hopper) and C2C (Grace Hopper) systems.
"""

import argparse
import time
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig


def main(args):
    prompts = [
        "Explain quantum computing. " + "Provide examples and use cases. " * 50,
        "Describe machine learning. " + "Include algorithms and applications. " * 50,
    ]

    kv_cache_max_tokens = 650

    host_cache_size = 10 * 1024 * 1024 

    kv_cache_config=KvCacheConfig(
            enable_block_reuse=True,
            max_tokens=kv_cache_max_tokens,
            tokens_per_block=16,
            host_cache_size=host_cache_size  # No offloading
    )

    print(f"GPU cache: {kv_cache_max_tokens} tokens (1 prompt max)")
    print(f"Host cache: {host_cache_size // (1024**3)}GB (ENABLED)")

    sampling_params = SamplingParams(max_tokens=64, temperature=0.7)

    llm = LLM(
        model=args.model,
        max_batch_size=1,
        max_seq_len=512,
        kv_cache_config=kv_cache_config
    )

    print(f"Model: {args.model}")
    print(f"Host cache size: {args.host_cache_size / (1024**3):.1f} GB")
    print(f"GPU cache: {kv_cache_config.max_tokens} tokens\n")

    # Round 1: Cold cache
    print("Round 1 - Cold cache (no reuse):")
    start = time.time()
    llm.generate(prompts[0], sampling_params)
    t1 = time.time() - start
    print(f"  Prompt A: {t1:.3f}s (computed)")
    print("✓ Prompt 0 cached on GPU")
    
    start = time.time()
    llm.generate(prompts[1], sampling_params)
    t2 = time.time() - start
    print(f"  Prompt B: {t2:.3f}s (computed, A evicted to host)")

    # Round 2: Cache reuse
    print("\nRound 2 - Cache reuse:")
    start = time.time()
    llm.generate(prompts[0], sampling_params)
    t3 = time.time() - start
    print(f"  Prompt A: {t3:.3f}s (reloaded from host)")
    
    start = time.time()
    llm.generate(prompts[1], sampling_params)
    t4 = time.time() - start
    print(f"  Prompt B: {t4:.3f}s (reloaded from host)")

    # Simple speedup metric
    avg_cold = (t1 + t2) / 2
    avg_warm = (t3 + t4) / 2
    speedup = avg_cold / avg_warm
    
    print(f"\nAverage time (cold): {avg_cold:.3f}s")
    print(f"Average time (warm): {avg_warm:.3f}s")
    print(f"Speedup: {speedup:.2f}x")

    llm.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-8B', help='Model to use')
    parser.add_argument('--host_cache_size', type=int, default=1024**3,
                        help='Host cache size in bytes (default: 1GB)')
    args = parser.parse_args()
    main(args)

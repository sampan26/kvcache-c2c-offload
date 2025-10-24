"""
Benchmark: KV Cache Offloading Performance with Accurate TTFT
Hopper (PCIe Gen5: 64 GB/s) vs Grace Hopper (C2C: 900 GB/s)

Measures accurate TTFT using async streaming API.
"""

import argparse
import time
import asyncio
import statistics
from typing import List, Dict
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig


class BenchmarkResult:
    def __init__(self, name: str):
        self.name = name
        self.cold_ttft: List[float] = []
        self.warm_ttft: List[float] = []
        self.cold_total: List[float] = []
        self.warm_total: List[float] = []
        
    def add_cold(self, ttft: float, total_time: float):
        self.cold_ttft.append(ttft)
        self.cold_total.append(total_time)
    
    def add_warm(self, ttft: float, total_time: float):
        self.warm_ttft.append(ttft)
        self.warm_total.append(total_time)
    
    def get_stats(self) -> Dict:
        return {
            'cold_ttft_avg': statistics.mean(self.cold_ttft) * 1000,
            'warm_ttft_avg': statistics.mean(self.warm_ttft) * 1000,
            'cold_total_avg': statistics.mean(self.cold_total) * 1000,
            'warm_total_avg': statistics.mean(self.warm_total) * 1000,
            'ttft_improvement_ms': (statistics.mean(self.cold_ttft) - statistics.mean(self.warm_ttft)) * 1000,
            'ttft_speedup': statistics.mean(self.cold_ttft) / statistics.mean(self.warm_ttft),
            'total_speedup': statistics.mean(self.cold_total) / statistics.mean(self.warm_total)
        }


async def measure_with_streaming(llm, prompt: str, sampling_params):
    """Measure accurate TTFT using async streaming API."""
    start_time = time.time()
    ttft = None
    
    async for output in llm.generate_async(prompt, sampling_params, streaming=True):
        if ttft is None:
            ttft = time.time() - start_time
    
    total_time = time.time() - start_time
    return ttft, total_time


async def run_benchmark_async(llm, prompts: List[str], sampling_params, num_iterations: int) -> BenchmarkResult:
    """Run benchmark with accurate TTFT measurement."""
    result = BenchmarkResult("benchmark")
    
    # Warmup
    for prompt in prompts:
        _ = llm.generate(prompt, sampling_params)
    
    # Benchmark: alternating A, B, A, B pattern
    for i in range(num_iterations):
        for prompt in prompts:
            ttft, total_time = await measure_with_streaming(llm, prompt, sampling_params)
            
            if i == 0:
                result.add_cold(ttft, total_time)
            else:
                result.add_warm(ttft, total_time)
    
    return result


def print_results(result: BenchmarkResult, system_name: str):
    """Print formatted benchmark results."""
    stats = result.get_stats()
    
    print(f"\n{'='*70}")
    print(f"Results: {system_name}")
    print(f"{'='*70}")
    
    print(f"\nTime to First Token (TTFT):")
    print(f"  Cold cache: {stats['cold_ttft_avg']:.1f} ms")
    print(f"  Warm cache: {stats['warm_ttft_avg']:.1f} ms")
    print(f"  TTFT improvement: {stats['ttft_improvement_ms']:.1f} ms ({stats['ttft_speedup']:.2f}x faster)")
    
    print(f"\nTotal Generation Time:")
    print(f"  Cold cache: {stats['cold_total_avg']:.1f} ms")
    print(f"  Warm cache: {stats['warm_total_avg']:.1f} ms")
    print(f"  Overall speedup: {stats['total_speedup']:.2f}x")
    
    total_requests = len(result.cold_ttft) + len(result.warm_ttft)
    total_time = sum(result.cold_total) + sum(result.warm_total)
    throughput = total_requests / total_time
    print(f"\nThroughput: {throughput:.2f} req/s")


def compare_systems(hopper_stats: Dict, grace_stats: Dict):
    """Compare Hopper vs Grace Hopper."""
    print(f"\n{'='*70}")
    print("HOPPER vs GRACE HOPPER COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<45} {'Hopper':<15} {'Grace Hopper':<15}")
    print(f"{'-'*70}")
    
    hopper_ttft = hopper_stats['warm_ttft_avg']
    grace_ttft = grace_stats['warm_ttft_avg']
    ttft_improvement = ((hopper_ttft - grace_ttft) / hopper_ttft) * 100
    
    print(f"{'TTFT (warm cache)':<45} {hopper_ttft:>8.1f} ms   {grace_ttft:>8.1f} ms")
    print(f"{'TTFT improvement':<45} {'':<15} {ttft_improvement:>8.1f}%")
    
    hopper_total = hopper_stats['warm_total_avg']
    grace_total = grace_stats['warm_total_avg']
    total_improvement = ((hopper_total - grace_total) / hopper_total) * 100
    
    print(f"\n{'Total time (warm cache)':<45} {hopper_total:>8.1f} ms   {grace_total:>8.1f} ms")
    print(f"{'Total improvement':<45} {'':<15} {total_improvement:>8.1f}%")
    
    hopper_tput = 1000 / hopper_total
    grace_tput = 1000 / grace_total
    tput_ratio = grace_tput / hopper_tput
    
    print(f"\n{'Throughput':<45} {hopper_tput:>8.2f} r/s   {grace_tput:>8.2f} r/s")
    print(f"{'Throughput ratio':<45} {'':<15} {tput_ratio:>8.2f}x")
    
    print(f"\n{'='*70}")
    print("KEY INSIGHT:")
    print(f"Grace Hopper's 900 GB/s C2C interconnect delivers:")
    print(f"  - {ttft_improvement:.0f}% faster TTFT (cache reload)")
    print(f"  - {tput_ratio:.2f}x higher throughput")
    print(f"  vs Hopper's 64 GB/s PCIe Gen5")
    print(f"{'='*70}\n")


async def main_async(args):
    prompts = [
        "Explain quantum computing. " + "Provide examples and use cases. " * 50,
        "Describe machine learning. " + "Include algorithms and applications. " * 50,
    ]

    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        max_tokens=650,
        tokens_per_block=16,
        host_cache_size=args.host_cache_size
    )
    
    sampling_params = SamplingParams(max_tokens=64, temperature=0.7)
    
    print(f"{'='*70}")
    print(f"BENCHMARK CONFIGURATION")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Host cache: {args.host_cache_size / (1024**3):.1f} GB")
    print(f"GPU cache: {kv_cache_config.max_tokens} tokens")
    print(f"Iterations: {args.iterations}")
    print(f"System: {args.system_name}")
    print(f"{'='*70}")
    
    llm = LLM(
        model=args.model,
        max_batch_size=1,
        max_seq_len=512,
        kv_cache_config=kv_cache_config
    )
    
    print("\nRunning benchmark with accurate TTFT measurement...")
    result = await run_benchmark_async(llm, prompts, sampling_params, args.iterations)
    
    print_results(result, args.system_name)
    
    llm.shutdown()
    
    if args.save_results:
        import json
        stats = result.get_stats()
        stats['system_name'] = args.system_name
        
        filename = f"benchmark_{args.system_name.lower().replace(' ', '_')}.json"
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nResults saved to {filename}")
    
    return result.get_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark KV cache offloading with accurate TTFT")
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--host_cache_size', type=int, default=10*1024**3, help='Host cache size in bytes')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('--system_name', default='Unknown System', help='System name for results')
    parser.add_argument('--save_results', action='store_true', help='Save results to JSON')
    parser.add_argument('--compare_with', type=str, help='Compare with another benchmark JSON')
    
    args = parser.parse_args()
    
    if args.compare_with:
        import json
        with open(args.compare_with, 'r') as f:
            other_stats = json.load(f)
        
        current_stats = asyncio.run(main_async(args))
        compare_systems(other_stats, current_stats)
    else:
        asyncio.run(main_async(args))
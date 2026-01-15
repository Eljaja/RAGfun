#!/usr/bin/env python3
"""
Advanced benchmarking script to find rustfs upload capacity limits.
Tests different worker counts, measures throughput, latency, and resource usage.
"""

import argparse
import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    """Result of a single upload operation"""
    key: str
    success: bool
    elapsed_seconds: float
    presign_time: float
    upload_time: float
    file_size_bytes: int
    error: Optional[str] = None


@dataclass
class BenchmarkRun:
    """Results from a single benchmark run"""
    worker_count: int
    upload_count: int
    file_size_bytes: int
    total_time: float
    successful_uploads: int
    failed_uploads: int
    avg_upload_time: float
    median_upload_time: float
    p95_upload_time: float
    p99_upload_time: float
    min_upload_time: float
    max_upload_time: float
    throughput_uploads_per_sec: float
    throughput_mbps: float
    avg_presign_time: float
    avg_actual_upload_time: float


class BenchmarkUploader:
    """Enhanced uploader with detailed timing metrics"""
    
    def __init__(self, base_url: str, bucket: str):
        self.base_url = base_url.rstrip('/')
        self.bucket = bucket
        self.presign_endpoint = f"{self.base_url}/api/objects/presign/put"
    
    def upload_with_metrics(
        self,
        file_path: Path,
        key: str,
        content_type: Optional[str] = None
    ) -> UploadResult:
        """Upload file and collect detailed timing metrics"""
        start_time = time.time()
        file_size = file_path.stat().st_size
        
        try:
            # Time the presign request
            presign_start = time.time()
            payload = {
                "bucket": self.bucket,
                "key": key,
                "expires_seconds": 300,
            }
            if content_type:
                payload["content_type"] = content_type
            
            resp = requests.post(self.presign_endpoint, json=payload, timeout=30)
            resp.raise_for_status()
            presigned_url = resp.json()["url"]
            presign_time = time.time() - presign_start
            
            # Time the actual upload
            upload_start = time.time()
            headers = {}
            if content_type:
                headers["Content-Type"] = content_type
            
            with open(file_path, 'rb') as f:
                upload_resp = requests.put(presigned_url, data=f, headers=headers, timeout=60)
                upload_resp.raise_for_status()
            upload_time = time.time() - upload_start
            
            elapsed = time.time() - start_time
            
            return UploadResult(
                key=key,
                success=True,
                elapsed_seconds=elapsed,
                presign_time=presign_time,
                upload_time=upload_time,
                file_size_bytes=file_size,
                error=None
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            return UploadResult(
                key=key,
                success=False,
                elapsed_seconds=elapsed,
                presign_time=0,
                upload_time=0,
                file_size_bytes=file_size,
                error=str(e)
            )


def run_benchmark(
    uploader: BenchmarkUploader,
    file_path: Path,
    num_uploads: int,
    num_workers: int,
    key_prefix: str,
    content_type: Optional[str] = None
) -> BenchmarkRun:
    """Run a single benchmark with specified worker count"""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"BENCHMARK: {num_workers} workers, {num_uploads} uploads")
    logger.info(f"{'='*70}")
    
    file_size = file_path.stat().st_size
    file_ext = file_path.suffix
    results = []
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        # Submit all tasks
        for i in range(num_uploads):
            key = f"{key_prefix}/worker{num_workers:03d}_file{i:05d}{file_ext}"
            future = executor.submit(
                uploader.upload_with_metrics,
                file_path=file_path,
                key=key,
                content_type=content_type
            )
            futures.append(future)
        
        # Collect results
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            if completed % 100 == 0:
                logger.info(f"Progress: {completed}/{num_uploads} uploads completed")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]
    
    if successful_results:
        upload_times = [r.elapsed_seconds for r in successful_results]
        presign_times = [r.presign_time for r in successful_results]
        actual_upload_times = [r.upload_time for r in successful_results]
        
        avg_upload_time = statistics.mean(upload_times)
        median_upload_time = statistics.median(upload_times)
        min_upload_time = min(upload_times)
        max_upload_time = max(upload_times)
        
        # Calculate percentiles
        sorted_times = sorted(upload_times)
        p95_idx = int(len(sorted_times) * 0.95)
        p99_idx = int(len(sorted_times) * 0.99)
        p95_upload_time = sorted_times[p95_idx] if p95_idx < len(sorted_times) else sorted_times[-1]
        p99_upload_time = sorted_times[p99_idx] if p99_idx < len(sorted_times) else sorted_times[-1]
        
        throughput_uploads = len(successful_results) / total_time
        throughput_mbps = (file_size * len(successful_results)) / (total_time * 1024 * 1024)
        
        avg_presign = statistics.mean(presign_times)
        avg_actual_upload = statistics.mean(actual_upload_times)
    else:
        avg_upload_time = median_upload_time = p95_upload_time = p99_upload_time = 0
        min_upload_time = max_upload_time = 0
        throughput_uploads = throughput_mbps = 0
        avg_presign = avg_actual_upload = 0
    
    # Print results
    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS: {num_workers} workers")
    logger.info(f"{'='*70}")
    logger.info(f"Total uploads: {len(results)}")
    logger.info(f"Successful: {len(successful_results)}")
    logger.info(f"Failed: {len(failed_results)}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"")
    logger.info(f"Timing (seconds):")
    logger.info(f"  Average: {avg_upload_time:.3f}s")
    logger.info(f"  Median:  {median_upload_time:.3f}s")
    logger.info(f"  Min:     {min_upload_time:.3f}s")
    logger.info(f"  Max:     {max_upload_time:.3f}s")
    logger.info(f"  P95:     {p95_upload_time:.3f}s")
    logger.info(f"  P99:     {p99_upload_time:.3f}s")
    logger.info(f"")
    logger.info(f"Breakdown:")
    logger.info(f"  Avg presign time: {avg_presign:.3f}s")
    logger.info(f"  Avg upload time:  {avg_actual_upload:.3f}s")
    logger.info(f"")
    logger.info(f"Throughput:")
    logger.info(f"  {throughput_uploads:.2f} uploads/second")
    logger.info(f"  {throughput_mbps:.2f} MB/s")
    logger.info(f"{'='*70}\n")
    
    # Print errors if any
    if failed_results:
        logger.warning(f"\nErrors encountered:")
        error_counts = {}
        for r in failed_results:
            error_counts[r.error] = error_counts.get(r.error, 0) + 1
        for error, count in error_counts.items():
            logger.warning(f"  {count}x: {error}")
    
    return BenchmarkRun(
        worker_count=num_workers,
        upload_count=num_uploads,
        file_size_bytes=file_size,
        total_time=total_time,
        successful_uploads=len(successful_results),
        failed_uploads=len(failed_results),
        avg_upload_time=avg_upload_time,
        median_upload_time=median_upload_time,
        p95_upload_time=p95_upload_time,
        p99_upload_time=p99_upload_time,
        min_upload_time=min_upload_time,
        max_upload_time=max_upload_time,
        throughput_uploads_per_sec=throughput_uploads,
        throughput_mbps=throughput_mbps,
        avg_presign_time=avg_presign,
        avg_actual_upload_time=avg_actual_upload
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark rustfs upload capacity with varying worker counts"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(__file__).parent / "file.pdf",
        help="Path to file to upload"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of presign service"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="test-bucket",
        help="S3 bucket name"
    )
    parser.add_argument(
        "--uploads-per-test",
        type=int,
        default=1000,
        help="Number of uploads per benchmark run"
    )
    parser.add_argument(
        "--worker-counts",
        type=str,
        default="1,5,10,20,50,100",
        help="Comma-separated list of worker counts to test"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="benchmark",
        help="Prefix for object keys"
    )
    parser.add_argument(
        "--content-type",
        type=str,
        default="application/pdf",
        help="Content type"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    # Validate file
    if not args.file.exists():
        logger.error(f"File not found: {args.file}")
        return 1
    
    # Parse worker counts
    worker_counts = [int(w.strip()) for w in args.worker_counts.split(',')]
    
    logger.info(f"\n{'='*70}")
    logger.info(f"BENCHMARK CONFIGURATION")
    logger.info(f"{'='*70}")
    logger.info(f"File: {args.file}")
    logger.info(f"File size: {args.file.stat().st_size / (1024*1024):.2f} MB")
    logger.info(f"Service: {args.url}")
    logger.info(f"Bucket: {args.bucket}")
    logger.info(f"Uploads per test: {args.uploads_per_test}")
    logger.info(f"Worker counts: {worker_counts}")
    logger.info(f"{'='*70}\n")
    
    # Create uploader
    uploader = BenchmarkUploader(base_url=args.url, bucket=args.bucket)
    
    # Run benchmarks
    all_results = []
    for worker_count in worker_counts:
        result = run_benchmark(
            uploader=uploader,
            file_path=args.file,
            num_uploads=args.uploads_per_test,
            num_workers=worker_count,
            key_prefix=args.prefix,
            content_type=args.content_type
        )
        all_results.append(result)
        
        # Short pause between runs
        time.sleep(2)
    
    # Print comparison table
    logger.info(f"\n{'='*70}")
    logger.info(f"BENCHMARK COMPARISON")
    logger.info(f"{'='*70}")
    logger.info(f"{'Workers':<10} {'Uploads/s':<12} {'MB/s':<10} {'Avg Time':<12} {'P99 Time':<12}")
    logger.info(f"{'-'*70}")
    
    for result in all_results:
        logger.info(
            f"{result.worker_count:<10} "
            f"{result.throughput_uploads_per_sec:<12.2f} "
            f"{result.throughput_mbps:<10.2f} "
            f"{result.avg_upload_time:<12.3f} "
            f"{result.p99_upload_time:<12.3f}"
        )
    logger.info(f"{'='*70}\n")
    
    # Find optimal worker count
    best_result = max(all_results, key=lambda r: r.throughput_uploads_per_sec)
    logger.info(f"🏆 BEST PERFORMANCE: {best_result.worker_count} workers")
    logger.info(f"   Throughput: {best_result.throughput_uploads_per_sec:.2f} uploads/s")
    logger.info(f"   Bandwidth: {best_result.throughput_mbps:.2f} MB/s")
    logger.info(f"   P99 Latency: {best_result.p99_upload_time:.3f}s\n")
    
    # Save results to JSON if requested
    if args.output:
        output_data = {
            "config": {
                "file": str(args.file),
                "file_size_bytes": args.file.stat().st_size,
                "service_url": args.url,
                "bucket": args.bucket,
                "uploads_per_test": args.uploads_per_test
            },
            "results": [asdict(r) for r in all_results]
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())



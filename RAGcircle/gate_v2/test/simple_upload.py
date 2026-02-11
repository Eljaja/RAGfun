#!/usr/bin/env python3
"""
Simple example script to upload file.pdf multiple times.
Quick and easy to use without many command-line options.
"""

import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
BASE_URL = "http://localhost:8000"
BUCKET = "test-bucket"
FILE_PATH = Path(__file__).parent / "file.pdf"
NUM_UPLOADS = 10
MAX_WORKERS = 5
KEY_PREFIX = "test"


def upload_once(file_path: Path, key: str, upload_num: int) -> dict:
    """Upload a file once using presigned PUT URL."""
    start = time.time()
    
    try:
        # Step 1: Get presigned URL
        presign_url = f"{BASE_URL}/api/objects/presign/put"
        payload = {
            "bucket": BUCKET,
            "key": key,
            "expires_seconds": 300,
            "content_type": "application/pdf"
        }
        
        print(f"[{upload_num:02d}] Requesting presigned URL for: {key}")
        resp = requests.post(presign_url, json=payload)
        resp.raise_for_status()
        
        presigned_data = resp.json()
        presigned_url = presigned_data["url"]
        
        # Step 2: Upload file using presigned URL
        print(f"[{upload_num:02d}] Uploading to presigned URL...")
        with open(file_path, 'rb') as f:
            upload_resp = requests.put(
                presigned_url, 
                data=f,
                headers={"Content-Type": "application/pdf"}
            )
            upload_resp.raise_for_status()
        
        elapsed = time.time() - start
        print(f"[{upload_num:02d}] ✓ Success! Uploaded {key} in {elapsed:.2f}s")
        
        return {"key": key, "success": True, "time": elapsed, "error": None}
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"[{upload_num:02d}] ✗ Failed: {e}")
        return {"key": key, "success": False, "time": elapsed, "error": str(e)}


def main():
    """Main function to upload file multiple times in parallel."""
    
    if not FILE_PATH.exists():
        print(f"Error: File not found: {FILE_PATH}")
        return
    
    print("=" * 60)
    print(f"File: {FILE_PATH}")
    print(f"Size: {FILE_PATH.stat().st_size} bytes")
    print(f"Service: {BASE_URL}")
    print(f"Bucket: {BUCKET}")
    print(f"Uploads: {NUM_UPLOADS}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print("=" * 60)
    print()
    
    start_time = time.time()
    results = []
    
    # Upload in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        
        # Submit all upload tasks
        for i in range(NUM_UPLOADS):
            key = f"{KEY_PREFIX}/file_{i:04d}.pdf"
            future = executor.submit(upload_once, FILE_PATH, key, i + 1)
            futures.append(future)
        
        # Wait for all to complete
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    total_time = time.time() - start_time
    
    # Print summary
    successes = sum(1 for r in results if r["success"])
    failures = len(results) - successes
    avg_time = sum(r["time"] for r in results) / len(results)
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {len(results)}")
    print(f"Success: {successes}")
    print(f"Failed: {failures}")
    print(f"Average time: {avg_time:.2f}s per upload")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {len(results) / total_time:.2f} uploads/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()



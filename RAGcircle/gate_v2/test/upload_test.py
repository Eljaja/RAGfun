#!/usr/bin/env python3
"""
Script to test presigned PUT URL uploads with parallel execution.
Uploads the same file multiple times with different names.
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PresignUploader:
    """Handles presigned URL generation and file uploads"""
    
    def __init__(self, base_url: str, bucket: str):
        """
        Initialize the uploader.
        
        Args:
            base_url: Base URL of the presign service (e.g., http://localhost:8000)
            bucket: S3 bucket name
        """
        self.base_url = base_url.rstrip('/')
        self.bucket = bucket
        self.presign_endpoint = f"{self.base_url}/api/objects/presign/put"
    
    def get_presigned_url(
        self, 
        key: str, 
        expires_seconds: int = 300,
        content_type: Optional[str] = None,
        max_size_bytes: Optional[int] = None
    ) -> tuple[str, str]:
        """
        Get a presigned PUT URL from the server.
        
        Args:
            key: Object key (filename) in S3
            expires_seconds: URL expiration time in seconds
            content_type: Content type of the file
            max_size_bytes: Maximum file size in bytes
            
        Returns:
            Tuple of (method, presigned_url)
        """
        payload = {
            "bucket": self.bucket,
            "key": key,
            "expires_seconds": expires_seconds,
        }
        
        if content_type:
            payload["content_type"] = content_type
        if max_size_bytes:
            payload["max_size_bytes"] = max_size_bytes
        
        logger.debug(f"Requesting presigned URL for key: {key}")
        response = requests.post(self.presign_endpoint, json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data["method"], data["url"]
    
    def upload_file(self, presigned_url: str, file_path: Path, content_type: Optional[str] = None) -> bool:
        """
        Upload a file using a presigned PUT URL.
        
        Args:
            presigned_url: The presigned URL to upload to
            file_path: Path to the file to upload
            content_type: Content type of the file
            
        Returns:
            True if upload was successful, False otherwise
        """
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type
        
        with open(file_path, 'rb') as f:
            response = requests.put(presigned_url, data=f, headers=headers)
            response.raise_for_status()
            return True
    
    def upload_with_presign(
        self, 
        file_path: Path, 
        key: str, 
        content_type: Optional[str] = None,
        max_size_bytes: Optional[int] = None
    ) -> dict:
        """
        Complete upload workflow: get presigned URL and upload file.
        
        Args:
            file_path: Path to the file to upload
            key: Object key (filename) in S3
            content_type: Content type of the file
            max_size_bytes: Maximum file size in bytes
            
        Returns:
            Dict with upload result information
        """
        start_time = time.time()
        
        try:
            # Step 1: Get presigned URL
            method, presigned_url = self.get_presigned_url(
                key=key,
                content_type=content_type,
                max_size_bytes=max_size_bytes
            )
            
            # Step 2: Upload file
            success = self.upload_file(presigned_url, file_path, content_type)
            
            elapsed = time.time() - start_time
            
            return {
                "key": key,
                "success": success,
                "elapsed_seconds": elapsed,
                "error": None
            }
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed to upload {key}: {e}")
            return {
                "key": key,
                "success": False,
                "elapsed_seconds": elapsed,
                "error": str(e)
            }


def upload_multiple_parallel(
    uploader: PresignUploader,
    file_path: Path,
    num_uploads: int,
    max_workers: int = 5,
    key_prefix: str = "test",
    content_type: Optional[str] = None,
    max_size_bytes: Optional[int] = None
) -> list[dict]:
    """
    Upload the same file multiple times with different keys in parallel.
    
    Args:
        uploader: PresignUploader instance
        file_path: Path to the file to upload
        num_uploads: Number of times to upload the file
        max_workers: Maximum number of parallel workers
        key_prefix: Prefix for the object keys
        content_type: Content type of the file
        max_size_bytes: Maximum file size in bytes
        
    Returns:
        List of upload results
    """
    logger.info(f"Starting {num_uploads} parallel uploads with {max_workers} workers")
    
    results = []
    file_ext = file_path.suffix
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all upload tasks
        future_to_key = {}
        for i in range(num_uploads):
            key = f"{key_prefix}/{file_path.stem}_{i:04d}{file_ext}"
            future = executor.submit(
                uploader.upload_with_presign,
                file_path=file_path,
                key=key,
                content_type=content_type,
                max_size_bytes=max_size_bytes
            )
            future_to_key[future] = key
        
        # Collect results as they complete
        for future in as_completed(future_to_key):
            result = future.result()
            results.append(result)
            
            if result["success"]:
                logger.info(
                    f"✓ Uploaded {result['key']} in {result['elapsed_seconds']:.2f}s"
                )
            else:
                logger.error(
                    f"✗ Failed {result['key']}: {result['error']}"
                )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Upload a file multiple times using presigned PUT URLs"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(__file__).parent / "file.pdf",
        help="Path to the file to upload (default: file.pdf in script directory)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the presign service (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="test-bucket",
        help="S3 bucket name (default: test-bucket)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of times to upload the file (default: 10)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="test",
        help="Prefix for object keys (default: test)"
    )
    parser.add_argument(
        "--content-type",
        type=str,
        default="application/pdf",
        help="Content type of the file (default: application/pdf)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Maximum file size in bytes (optional)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate file exists
    if not args.file.exists():
        logger.error(f"File not found: {args.file}")
        return 1
    
    logger.info(f"File: {args.file} ({args.file.stat().st_size} bytes)")
    logger.info(f"Service URL: {args.url}")
    logger.info(f"Bucket: {args.bucket}")
    logger.info(f"Uploads: {args.count}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Key prefix: {args.prefix}")
    
    # Create uploader
    uploader = PresignUploader(base_url=args.url, bucket=args.bucket)
    
    # Perform uploads
    start_time = time.time()
    results = upload_multiple_parallel(
        uploader=uploader,
        file_path=args.file,
        num_uploads=args.count,
        max_workers=args.workers,
        key_prefix=args.prefix,
        content_type=args.content_type,
        max_size_bytes=args.max_size
    )
    total_time = time.time() - start_time
    
    # Print summary
    successes = sum(1 for r in results if r["success"])
    failures = len(results) - successes
    avg_time = sum(r["elapsed_seconds"] for r in results) / len(results) if results else 0
    
    logger.info("=" * 60)
    logger.info("UPLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total uploads: {len(results)}")
    logger.info(f"Successful: {successes}")
    logger.info(f"Failed: {failures}")
    logger.info(f"Average time per upload: {avg_time:.2f}s")
    logger.info(f"Total execution time: {total_time:.2f}s")
    logger.info(f"Throughput: {len(results) / total_time:.2f} uploads/second")
    logger.info("=" * 60)
    
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    exit(main())


#!/usr/bin/env python3
"""
Script to test multipart document uploads with parallel execution.
Uploads the same file multiple times with different doc_ids.
"""

import argparse
import logging
import time
import uuid
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


class MultipartUploader:
    """Handles multipart document uploads"""
    
    def __init__(self, base_url: str):
        """
        Initialize the uploader.
        
        Args:
            base_url: Base URL of the document service (e.g., http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        self.upload_endpoint = f"{self.base_url}/v1/documents/upload"
    
    def upload_document(
        self,
        file_path: Path,
        doc_id: str,
        title: Optional[str] = None,
        uri: Optional[str] = None,
        source: Optional[str] = None,
        lang: Optional[str] = None,
        tags: Optional[str] = None,
        acl: Optional[str] = None,
        tenant_id: Optional[str] = None,
        project_id: Optional[str] = None,
        refresh: bool = False
    ) -> dict:
        """
        Upload a document using multipart form data.
        
        Args:
            file_path: Path to the file to upload
            doc_id: Document ID (required)
            title: Document title
            uri: Document URI
            source: Document source
            lang: Document language
            tags: Comma-separated tags (e.g., "tag1,tag2")
            acl: Comma-separated ACL entries
            tenant_id: Tenant ID
            project_id: Project ID (used as collection)
            refresh: Whether to refresh after upload
            
        Returns:
            Dict with upload result information
        """
        start_time = time.time()
        
        try:
            # Prepare form data
            data = {
                "doc_id": doc_id,
                "refresh": str(refresh).lower()
            }
            
            if title:
                data["title"] = title
            if uri:
                data["uri"] = uri
            if source:
                data["source"] = source
            if lang:
                data["lang"] = lang
            if tags:
                data["tags"] = tags
            if acl:
                data["acl"] = acl
            if tenant_id:
                data["tenant_id"] = tenant_id
            if project_id:
                data["project_id"] = project_id
            
            # Prepare file
            with open(file_path, 'rb') as f:
                files = {
                    "file": (file_path.name, f, self._guess_content_type(file_path))
                }
                
                logger.debug(f"Uploading document with doc_id: {doc_id}")
                response = requests.post(
                    self.upload_endpoint,
                    data=data,
                    files=files
                )
                response.raise_for_status()
            
            elapsed = time.time() - start_time
            
            return {
                "doc_id": doc_id,
                "success": True,
                "elapsed_seconds": elapsed,
                "error": None,
                "response": response.json() if response.content else None
            }
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed to upload doc_id {doc_id}: {e}")
            return {
                "doc_id": doc_id,
                "success": False,
                "elapsed_seconds": elapsed,
                "error": str(e),
                "response": None
            }
    
    def _guess_content_type(self, file_path: Path) -> str:
        """Guess content type based on file extension."""
        extension_map = {
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".html": "text/html",
            ".htm": "text/html",
            ".json": "application/json",
            ".xml": "application/xml",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".ppt": "application/vnd.ms-powerpoint",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".md": "text/markdown",
            ".csv": "text/csv",
        }
        return extension_map.get(file_path.suffix.lower(), "application/octet-stream")


def upload_multiple_parallel(
    uploader: MultipartUploader,
    file_path: Path,
    num_uploads: int,
    max_workers: int = 5,
    doc_id_prefix: str = "test",
    title: Optional[str] = None,
    uri: Optional[str] = None,
    source: Optional[str] = None,
    lang: Optional[str] = None,
    tags: Optional[str] = None,
    acl: Optional[str] = None,
    tenant_id: Optional[str] = None,
    project_id: Optional[str] = None,
    refresh: bool = False
) -> list[dict]:
    """
    Upload the same file multiple times with different doc_ids in parallel.
    
    Args:
        uploader: MultipartUploader instance
        file_path: Path to the file to upload
        num_uploads: Number of times to upload the file
        max_workers: Maximum number of parallel workers
        doc_id_prefix: Prefix for document IDs
        title: Document title
        uri: Document URI
        source: Document source
        lang: Document language
        tags: Comma-separated tags
        acl: Comma-separated ACL entries
        tenant_id: Tenant ID
        project_id: Project ID (collection)
        refresh: Whether to refresh after upload
        
    Returns:
        List of upload results
    """
    logger.info(f"Starting {num_uploads} parallel uploads with {max_workers} workers")
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all upload tasks
        future_to_doc_id = {}
        for i in range(num_uploads):
            doc_id = f"{doc_id_prefix}_{i:04d}_{uuid.uuid4().hex[:8]}"
            doc_title = f"{title}_{i:04d}" if title else None
            
            future = executor.submit(
                uploader.upload_document,
                file_path=file_path,
                doc_id=doc_id,
                title=doc_title,
                uri=uri,
                source=source,
                lang=lang,
                tags=tags,
                acl=acl,
                tenant_id=tenant_id,
                project_id=project_id,
                refresh=refresh
            )
            future_to_doc_id[future] = doc_id
        
        # Collect results as they complete
        for future in as_completed(future_to_doc_id):
            result = future.result()
            results.append(result)
            
            if result["success"]:
                logger.info(
                    f"✓ Uploaded {result['doc_id']} in {result['elapsed_seconds']:.2f}s"
                )
            else:
                logger.error(
                    f"✗ Failed {result['doc_id']}: {result['error']}"
                )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Upload a document multiple times using multipart form upload"
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
        help="Base URL of the document service (default: http://localhost:8000)"
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
        "--doc-id-prefix",
        type=str,
        default="test",
        help="Prefix for document IDs (default: test)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Document title (optional)"
    )
    parser.add_argument(
        "--uri",
        type=str,
        default=None,
        help="Document URI (optional)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Document source (optional)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Document language (optional)"
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated tags (optional, e.g., 'tag1,tag2')"
    )
    parser.add_argument(
        "--acl",
        type=str,
        default=None,
        help="Comma-separated ACL entries (optional)"
    )
    parser.add_argument(
        "--tenant-id",
        type=str,
        default=None,
        help="Tenant ID (optional)"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=None,
        help="Project ID / collection name (optional)"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh after upload (default: false)"
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
    logger.info(f"Uploads: {args.count}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Doc ID prefix: {args.doc_id_prefix}")
    if args.project_id:
        logger.info(f"Project ID (collection): {args.project_id}")
    if args.tenant_id:
        logger.info(f"Tenant ID: {args.tenant_id}")
    if args.tags:
        logger.info(f"Tags: {args.tags}")
    
    # Create uploader
    uploader = MultipartUploader(base_url=args.url)
    
    # Perform uploads
    start_time = time.time()
    results = upload_multiple_parallel(
        uploader=uploader,
        file_path=args.file,
        num_uploads=args.count,
        max_workers=args.workers,
        doc_id_prefix=args.doc_id_prefix,
        title=args.title,
        uri=args.uri,
        source=args.source,
        lang=args.lang,
        tags=args.tags,
        acl=args.acl,
        tenant_id=args.tenant_id,
        project_id=args.project_id,
        refresh=args.refresh
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

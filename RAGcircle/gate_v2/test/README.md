# Presigned URL Upload Test Scripts

This directory contains test scripts to upload files multiple times using presigned PUT URLs with parallel execution.

## Files

- **`file.pdf`** - Sample PDF file for testing
- **`upload_test.py`** - Full-featured upload script with command-line options
- **`simple_upload.py`** - Simple, easy-to-use upload script with hardcoded defaults
- **`requirements.txt`** - Python dependencies

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your presign service is running (default: `http://localhost:8000`)

3. Ensure you have a test bucket created (or the script will attempt to upload to it)

## Quick Start (Simple Script)

Edit the configuration variables in `simple_upload.py` if needed:

```python
BASE_URL = "http://localhost:8000"
BUCKET = "test-bucket"
NUM_UPLOADS = 10
MAX_WORKERS = 5
```

Then run:

```bash
python simple_upload.py
```

This will upload `file.pdf` 10 times with different names in parallel.

## Advanced Usage (Full Script)

The `upload_test.py` script provides full command-line control:

### Basic usage:
```bash
python upload_test.py --bucket test-bucket --count 20
```

### All available options:
```bash
python upload_test.py \
  --file file.pdf \
  --url http://localhost:8000 \
  --bucket my-bucket \
  --count 50 \
  --workers 10 \
  --prefix uploads \
  --content-type application/pdf \
  --max-size 104857600 \
  --verbose
```

### Options:

- `--file PATH` - Path to file to upload (default: `file.pdf`)
- `--url URL` - Base URL of presign service (default: `http://localhost:8000`)
- `--bucket NAME` - S3 bucket name (default: `test-bucket`)
- `--count N` - Number of times to upload (default: 10)
- `--workers N` - Parallel workers (default: 5)
- `--prefix PREFIX` - Object key prefix (default: `test`)
- `--content-type TYPE` - Content type (default: `application/pdf`)
- `--max-size BYTES` - Max file size in bytes (optional)
- `--verbose` - Enable debug logging

## How It Works

Both scripts follow this pipeline for each upload:

1. **Request presigned URL**: POST to `/api/objects/presign/put` with:
   ```json
   {
     "bucket": "test-bucket",
     "key": "test/file_0001.pdf",
     "expires_seconds": 300,
     "content_type": "application/pdf"
   }
   ```

2. **Receive presigned URL**: Get response:
   ```json
   {
     "method": "PUT",
     "url": "https://s3.amazonaws.com/...",
     "expires": 300
   }
   ```

3. **Upload file**: PUT the file data to the presigned URL

4. **Repeat**: Do steps 1-3 multiple times with different filenames

All uploads run in parallel using `concurrent.futures.ThreadPoolExecutor` for maximum efficiency.

## Example Output

```
============================================================
File: file.pdf (8945 bytes)
Service: http://localhost:8000
Bucket: test-bucket
Uploads: 10
Parallel workers: 5
============================================================

[01] Requesting presigned URL for: test/file_0000.pdf
[02] Requesting presigned URL for: test/file_0001.pdf
...
[01] ✓ Success! Uploaded test/file_0000.pdf in 0.45s
[02] ✓ Success! Uploaded test/file_0001.pdf in 0.48s
...

============================================================
SUMMARY
============================================================
Total: 10
Success: 10
Failed: 0
Average time: 0.46s per upload
Total time: 1.23s
Throughput: 8.13 uploads/sec
============================================================
```

## Troubleshooting

### Connection refused
Make sure the presign service is running on the specified URL.

### Bucket does not exist
Create the bucket first using the API or ensure it exists.

### Permission denied
Check your AWS/S3 credentials in the presign service configuration.

### File not found
Ensure `file.pdf` exists in the test directory or specify the correct path with `--file`.



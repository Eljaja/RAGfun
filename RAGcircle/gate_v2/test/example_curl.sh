#!/bin/bash
# Example curl command for uploading a file using a presigned PUT URL
# 
# Replace the URL with your actual presigned URL from the API response
# Replace the file path with your actual file path

# Basic curl command (if content_type was NOT specified when generating the presigned URL)
curl -X PUT \
  "http://localhost:9004/heheheh/file.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=rustfs%2F20260113%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20260113T155307Z&X-Amz-Expires=300&X-Amz-SignedHeaders=content-type%3Bhost&X-Amz-Signature=6554768dde2500acf8151cf9c7ebbc5cc860a750b1ca2e23bb2d1cbadf7eac84" \
  --upload-file test/file.pdf

# IMPORTANT: If content_type was specified when generating the presigned URL,
# you MUST include the Content-Type header and it MUST match exactly
curl -X PUT \
  -H "Content-Type: application/pdf" \
  "http://localhost:9004/heheheh/file.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=rustfs%2F20260113%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20260113T155307Z&X-Amz-Expires=300&X-Amz-SignedHeaders=content-type%3Bhost&X-Amz-Signature=6554768dde2500acf8151cf9c7ebbc5cc860a750b1ca2e23bb2d1cbadf7eac84" \
  --upload-file test/file.pdf

# Alternative: Using -T flag (shorthand for --upload-file)
curl -X PUT \
  -H "Content-Type: application/pdf" \
  -T test/file.pdf \
  "http://localhost:9004/heheheh/file.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=rustfs%2F20260113%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20260113T155307Z&X-Amz-Expires=300&X-Amz-SignedHeaders=content-type%3Bhost&X-Amz-Signature=6554768dde2500acf8151cf9c7ebbc5cc860a750b1ca2e23bb2d1cbadf7eac84"

# With verbose output for debugging
curl -v -X PUT \
  -H "Content-Type: application/pdf" \
  -T test/file.pdf \
  "http://localhost:9004/heheheh/file.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=rustfs%2F20260113%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20260113T155307Z&X-Amz-Expires=300&X-Amz-SignedHeaders=content-type%3Bhost&X-Amz-Signature=6554768dde2500acf8151cf9c7ebbc5cc860a750b1ca2e23bb2d1cbadf7eac84"


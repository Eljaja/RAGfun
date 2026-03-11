import boto3
from botocore.client import Config


# curl -X PUT \
#   -H "Content-Type: application/pdf" \
#   -T file.pdf \
#   "http://localhost:9004/heheheh/file.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=rustfs%2F20260113%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20260113T155307Z&X-Amz-Expires=300&X-Amz-SignedHeaders=content-type%3Bhost&X-Amz-Signature=6554768dde2500acf8151cf9c7ebbc5cc860a750b1ca2e23bb2d1cbadf7eac84"

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9004",   # <-- your published port
    aws_access_key_id="rustfs",             # <-- your RustFS creds
    aws_secret_access_key="password",
    # region_name="eu-central",
    config=Config(signature_version="s3v4"),
)

s3.create_bucket(Bucket='uploads5')

notification_config = {
    'QueueConfigurations': [{
        'Id': 'webhook-notification',
        'QueueArn': 'arn:rustfs:sqs:us-east-1:webhook:webhook',  # Reference the 'webhook' target
        'Events': ['s3:ObjectCreated:*', 's3:ObjectRemoved:*']
    }]
}

s3.put_bucket_notification_configuration(
    Bucket='uploads5',
    NotificationConfiguration=notification_config
)

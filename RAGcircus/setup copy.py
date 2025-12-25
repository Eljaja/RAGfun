import boto3
from botocore.client import Config

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9004",   # <-- your published port
    aws_access_key_id="rustfs",             # <-- your RustFS creds
    aws_secret_access_key="password",
    region_name="eu-central",
    config=Config(signature_version="s3v4"),
)

s3.create_bucket(Bucket='uploads5')

notification_config = {
    'QueueConfigurations': [{
        'Id': 'webhook-notification',
        'QueueArn': 'webhook:webhook',  # Reference the 'webhook' target
        'Events': ['s3:ObjectCreated:*', 's3:ObjectRemoved:*']
    }]
}

s3.put_bucket_notification_configuration(
    Bucket='uploads5',
    NotificationConfiguration=notification_config
)
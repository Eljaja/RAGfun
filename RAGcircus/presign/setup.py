# Upload with detailed logging
import boto3

s3 = boto3.client('s3', endpoint_url='http://localhost:9004',
                  aws_access_key_id='rustfs', aws_secret_access_key='password')

print("Uploading file...")
response = s3.put_object(Bucket='lolol', Key='test123.txt', Body=b'hello world')
print(f"Upload response: {response['ResponseMetadata']['HTTPStatusCode']}")

# Watch docker logs
# docker logs rustfs -f
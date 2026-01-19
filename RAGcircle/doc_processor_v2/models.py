from datetime import datetime
from typing import Literal, Any
from pydantic import BaseModel, Field, HttpUrl


class RustfsUserIdentity(BaseModel):
    principalId: str = ""


class RustfsRequestParameters(BaseModel):
    accept_encoding: str = Field(..., alias="accept-encoding")
    host: str
    user_agent: str = Field(..., alias="user-agent")
    content_type: str | None = Field(None, alias="content-type")
    accept: str
    x_request_id: str = Field(..., alias="x-request-id")
    connection: str
    content_length: str = Field(..., alias="content-length")


class RustfsResponseElements(BaseModel):
    x_amz_request_id: str = Field("", alias="x-amz-request-id")
    x_amz_id_2: str = Field("", alias="x-amz-id-2")


class RustfsS3Bucket(BaseModel):
    name: str
    ownerIdentity: dict[str, str] = Field(default_factory=lambda: {"principalId": ""})
    arn: str


class RustfsS3Object(BaseModel):
    key: str
    size: int
    etag: str
    contentType: str | None = None
    userMetadata: dict[str, str] = Field(default_factory=dict)
    versionId: str = ""
    sequencer: str


class RustfsS3(BaseModel):
    s3SchemaVersion: str
    configurationId: str
    bucket: RustfsS3Bucket
    object: RustfsS3Object


class RustfsSource(BaseModel):
    host: str
    port: str = ""
    userAgent: str


class RustfsEventData(BaseModel):
    eventVersion: str
    eventSource: Literal["rustfs:s3"]
    awsRegion: str = ""
    eventTime: datetime
    eventName: Literal["s3:ObjectCreated:Put", "s3:ObjectRemoved:Delete", ...]  # extend as needed
    userIdentity: RustfsUserIdentity
    requestParameters: RustfsRequestParameters
    responseElements: RustfsResponseElements
    s3: RustfsS3
    source: RustfsSource


class RustfsRecord(BaseModel):
    object_name: str
    bucket_name: str
    event_name: str
    data: RustfsEventData


class RustfsS3Event(BaseModel):
    """rustfs notification event format (2025-2026 observed)"""

    EventName: str = Field(..., pattern=r"^s3:")
    Key: str
    Records: list[RustfsRecord]


# Quick usage example:
def parse_rustfs_event(payload: dict[str, Any]) -> RustfsS3Event:
    model = RustfsS3Event.model_validate(payload)
    print("-------------------")
    print(model)
    return model

# Bonus one-liner style if you just want the juicy parts quickly:
def get_essentials(event: dict) -> dict:
    record = event["Records"][0]
    obj = record["data"]["s3"]["object"]
    return {
        "bucket": record["bucket_name"],
        "key": obj["key"],
        "size": obj["size"],
        "etag": obj["etag"],
        "content_type": obj.get("contentType"),
        "inline_data": obj["userMetadata"].get("x-rustfs-internal-inline-data") == "true",
        "event_time": record["data"]["eventTime"],
    }
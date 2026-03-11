from errors import NonRetryableError
from events import extract_s3_event_info


def test_extract_rustfs_record_shape():
    event = {
        "Records": [
            {
                "bucket_name": "b",
                "object_name": "k.pdf",
                "event_name": "s3:ObjectCreated:Put",
            }
        ]
    }
    info = extract_s3_event_info(event)
    assert info.bucket == "b"
    assert info.key == "k.pdf"
    assert "ObjectCreated" in info.event_name


def test_extract_aws_s3_shape():
    event = {
        "Records": [
            {
                "eventName": "ObjectRemoved:Delete",
                "s3": {"bucket": {"name": "b"}, "object": {"key": "a/b.txt"}},
            }
        ]
    }
    info = extract_s3_event_info(event)
    assert info.bucket == "b"
    assert info.key == "a/b.txt"
    assert "ObjectRemoved" in info.event_name


def test_extract_top_level_key_path():
    event = {"EventName": "ObjectCreated:Put", "Key": "bucket1/path/to/file.pdf"}
    info = extract_s3_event_info(event)
    assert info.bucket == "bucket1"
    assert info.key == "path/to/file.pdf"


def test_extract_malformed_raises_nonretryable():
    with pytest.raises(NonRetryableError):
        extract_s3_event_info({"foo": "bar"})


import pytest  # keep at end to avoid tooling reordering issues


from amqp_consumer import retry_count_from_headers


def test_retry_count_prefers_custom_header():
    assert retry_count_from_headers({"x-retry-count": 3}, retry_queue="q") == 3


def test_retry_count_falls_back_to_x_death():
    hdrs = {
        "x-death": [
            {"queue": "other", "reason": "expired", "count": 99},
            {"queue": "retry_q", "reason": "expired", "count": 2},
        ]
    }
    assert retry_count_from_headers(hdrs, retry_queue="retry_q") == 2


def test_retry_count_invalid_headers_is_zero():
    assert retry_count_from_headers(None, retry_queue="q") == 0
    assert retry_count_from_headers({"x-death": "nope"}, retry_queue="q") == 0


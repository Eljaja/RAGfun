from __future__ import annotations

import asyncio
import json

import aio_pika

from config import AppConfig
from errors import NonRetryableError
from events import extract_s3_event_info
from pipeline import PipelineDeps, handle_s3_event


def retry_count_from_headers(headers: dict | None, *, retry_queue: str) -> int:
    """
    Determine retry count.

    We track retry count in a custom header since we manually publish to retry queue.
    Also check x-death for messages that came back from the retry queue TTL expiry.
    """
    if not headers:
        return 0

    # Custom header (preferred)
    custom_count = headers.get("x-retry-count")
    if custom_count is not None:
        try:
            return int(custom_count)
        except Exception:
            pass

    # Fallback: RabbitMQ x-death
    deaths = headers.get("x-death")
    if not isinstance(deaths, list):
        return 0
    for d in deaths:
        if not isinstance(d, dict):
            continue
        if d.get("queue") == retry_queue and d.get("reason") == "expired":
            try:
                return int(d.get("count") or 0)
            except Exception:
                return 0
    return 0


async def publish_to_retry(
    *,
    message: aio_pika.IncomingMessage,
    retry_exchange: aio_pika.Exchange,
    routing_key: str,
    retry_count: int,
) -> None:
    hdrs = dict(message.headers or {})
    hdrs["x-retry-count"] = retry_count + 1
    msg = aio_pika.Message(
        body=message.body,
        headers=hdrs,
        content_type=message.content_type,
        content_encoding=message.content_encoding,
        correlation_id=message.correlation_id,
        message_id=message.message_id,
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
    )
    await retry_exchange.publish(msg, routing_key=routing_key)


async def publish_to_dlq(
    *,
    message: aio_pika.IncomingMessage,
    dlq_exchange: aio_pika.Exchange,
    routing_key: str,
    reason: str,
    retry_count: int,
) -> None:
    hdrs = dict(message.headers or {})
    hdrs["dlq_reason"] = reason
    hdrs["x-retry-count"] = retry_count
    msg = aio_pika.Message(
        body=message.body,
        headers=hdrs,
        content_type=message.content_type,
        content_encoding=message.content_encoding,
        correlation_id=message.correlation_id,
        message_id=message.message_id,
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
    )
    await dlq_exchange.publish(msg, routing_key=routing_key)


async def handle_incoming_message(
    *,
    message: aio_pika.IncomingMessage,
    s3_client,
    deps: PipelineDeps,
    retry_exchange: aio_pika.Exchange,
    dlq_exchange: aio_pika.Exchange,
    cfg: AppConfig,
) -> None:
    """
    Consume one message.

    Important reliability rule:
    - We only ACK after either (a) successful processing, or (b) successful publish to retry/DLQ.
    - If publish fails, we do NOT ACK; the message will be redelivered.
    """
    retry_count = retry_count_from_headers(message.headers, retry_queue=cfg.amqp_retry_queue)

    # Parse JSON payload (DLQ immediately if invalid)
    try:
        decoded = message.body.decode("utf-8")
        event = json.loads(decoded)
    except Exception as e:
        await publish_to_dlq(
            message=message,
            dlq_exchange=dlq_exchange,
            routing_key=cfg.amqp_dlq_routing_key,
            reason=f"invalid_json:{type(e).__name__}",
            retry_count=retry_count,
        )
        await message.ack()
        return

    if not isinstance(event, dict):
        await publish_to_dlq(
            message=message,
            dlq_exchange=dlq_exchange,
            routing_key=cfg.amqp_dlq_routing_key,
            reason="event_payload_not_a_dict",
            retry_count=retry_count,
        )
        await message.ack()
        return

    # Parse event shape (DLQ immediately if malformed)
    try:
        info = extract_s3_event_info(event)
    except NonRetryableError as e:
        await publish_to_dlq(
            message=message,
            dlq_exchange=dlq_exchange,
            routing_key=cfg.amqp_dlq_routing_key,
            reason=f"non_retryable:{e}",
            retry_count=retry_count,
        )
        await message.ack()
        return

    # Main pipeline
    try:
        await handle_s3_event(info=info, s3_client=s3_client, deps=deps)
        await message.ack()
        return
    except NonRetryableError as e:
        await publish_to_dlq(
            message=message,
            dlq_exchange=dlq_exchange,
            routing_key=cfg.amqp_dlq_routing_key,
            reason=f"non_retryable:{e}",
            retry_count=retry_count,
        )
        await message.ack()
        return
    except Exception as e:
        # Retryable failure
        if retry_count >= cfg.amqp_max_retries:
            await publish_to_dlq(
                message=message,
                dlq_exchange=dlq_exchange,
                routing_key=cfg.amqp_dlq_routing_key,
                reason=f"max_retries_exceeded:{type(e).__name__}:{e}",
                retry_count=retry_count,
            )
            await message.ack()
            return
        await publish_to_retry(
            message=message,
            retry_exchange=retry_exchange,
            routing_key=cfg.amqp_retry_routing_key,
            retry_count=retry_count,
        )
        await message.ack()
        return


async def consume_rabbitmq(*, s3_client, deps: PipelineDeps, cfg: AppConfig) -> None:
    """
    Async RabbitMQ consumer with reconnection logic.
    Declares retry+DLQ exchanges/queues (service-owned), and binds to the RustFS-managed queue.
    """
    while True:
        try:
            connection = await aio_pika.connect_robust(cfg.rabbitmq_url, heartbeat=30)
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=10)

            # DLQ wiring (parking lot)
            dlx = await channel.declare_exchange(cfg.amqp_dlx_exchange, aio_pika.ExchangeType.DIRECT, durable=True)
            dlq = await channel.declare_queue(cfg.amqp_dlq_queue, durable=True)
            await dlq.bind(exchange=dlx, routing_key=cfg.amqp_dlq_routing_key)

            # Retry wiring (TTL -> dead-letter back to main exchange/routing key)
            retry_ex = await channel.declare_exchange(cfg.amqp_retry_exchange, aio_pika.ExchangeType.DIRECT, durable=True)
            retry_q = await channel.declare_queue(
                cfg.amqp_retry_queue,
                durable=True,
                arguments={
                    "x-message-ttl": cfg.amqp_retry_ttl_ms,
                    "x-dead-letter-exchange": cfg.amqp_exchange,
                    "x-dead-letter-routing-key": cfg.amqp_binding_key,
                },
            )
            await retry_q.bind(exchange=retry_ex, routing_key=cfg.amqp_retry_routing_key)

            # Main queue: RustFS-managed; we bind to exchange+routing key
            queue = await channel.declare_queue(cfg.amqp_queue, durable=True)
            await queue.bind(exchange=cfg.amqp_exchange, routing_key=cfg.amqp_binding_key)

            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    await handle_incoming_message(
                        message=message,
                        s3_client=s3_client,
                        deps=deps,
                        retry_exchange=retry_ex,
                        dlq_exchange=dlx,
                        cfg=cfg,
                    )
        except asyncio.CancelledError:
            break
        except Exception:
            # basic reconnect loop; caller can add backoff/logging
            await asyncio.sleep(3)


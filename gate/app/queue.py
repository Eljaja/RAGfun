from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import aio_pika
from aio_pika import Message

logger = logging.getLogger("gate.queue")


DEFAULT_QUEUE = "ingestion.tasks"


@dataclass
class RabbitPublisher:
    url: str
    queue_name: str = DEFAULT_QUEUE

    _conn: aio_pika.RobustConnection | None = None
    _channel: aio_pika.RobustChannel | None = None
    _queue: aio_pika.RobustQueue | None = None

    async def start(self) -> None:
        self._conn = await aio_pika.connect_robust(self.url)
        self._channel = await self._conn.channel(publisher_confirms=False)
        await self._channel.set_qos(prefetch_count=10)
        self._queue = await self._channel.declare_queue(self.queue_name, durable=True)
        logger.info("rabbit_publisher_ready", extra={"extra": {"queue": self.queue_name}})

    async def close(self) -> None:
        try:
            if self._channel:
                await self._channel.close()
        finally:
            self._channel = None
            self._queue = None
        if self._conn:
            await self._conn.close()
        self._conn = None

    async def publish(self, *, payload: dict[str, Any], headers: dict[str, Any] | None = None) -> None:
        if not self._channel:
            raise RuntimeError("rabbit_not_started")
        body = json.dumps(payload).encode("utf-8")
        msg = Message(
            body=body,
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            headers=headers or {},
        )
        # Default exchange routes directly to queue by name.
        await self._channel.default_exchange.publish(msg, routing_key=self.queue_name)



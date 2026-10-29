"""Outbound notification hook.

The prototype only logs that a donor was contacted (donor id + request id, never
phone numbers or names). Replace ``send`` with an SMS / WhatsApp / push gateway
integration for a real deployment, using only approved communication channels.
"""
from __future__ import annotations

import logging

log = logging.getLogger("notifier")


def send(donor_id: int, request_id: int, urgency: str) -> None:
    log.info("notify donor_id=%s request_id=%s urgency=%s", donor_id, request_id, urgency)

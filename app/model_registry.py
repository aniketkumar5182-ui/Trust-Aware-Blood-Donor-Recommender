"""Lazy loader for the trained response model (optional - the API works without it)."""
from __future__ import annotations

import logging
from typing import Optional

from .config import MODEL_PATH
from .ml_model import ResponseModel

log = logging.getLogger(__name__)
_model: Optional[ResponseModel] = None
_loaded = False


def get_model(reload: bool = False) -> Optional[ResponseModel]:
    global _model, _loaded
    if reload or not _loaded:
        _loaded = True
        if MODEL_PATH.exists():
            try:
                _model = ResponseModel.load(MODEL_PATH)
            except Exception:  # corrupt / incompatible file -> fall back to weighted scoring
                log.exception("Could not load %s; using the weighted score only", MODEL_PATH)
                _model = None
        else:
            _model = None
    return _model

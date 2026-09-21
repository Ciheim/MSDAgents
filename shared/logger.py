from __future__ import annotations

import fcntl
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Any

import yaml

OUTPUT_LOG_FILE = "MSD_Chatbot_Log.yaml"
LOG_ROOT_TITLE = "MSD Chabot Log"
AI_MODEL_KEY = "AI model"
SYSTEM_PROMPT_KEY = "System prompt"
RETRIEVED_FILES_KEY = "retrieved files"
RETENTION_DAYS = 30

def configure_yaml_logging(log_file: str | Path = OUTPUT_LOG_FILE) -> Path:
    return Path(log_file)


def _normalize_timestamp(value: str) -> datetime | None:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return None

    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)

    return timestamp.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _load_log_root(existing_data: Any, model_name: str, system_prompt: str) -> dict[str, Any]:
    existing_root = existing_data.get(LOG_ROOT_TITLE, {})
    if not isinstance(existing_root, dict):
        existing_root = {}

    log_root: dict[str, Any] = {
        AI_MODEL_KEY: model_name,
        SYSTEM_PROMPT_KEY: system_prompt,
    }
    for key, value in existing_root.items():
        if key in {AI_MODEL_KEY, SYSTEM_PROMPT_KEY}:
            continue
        log_root[key] = value

    return log_root


def _prune_expired_entries(log_root: dict[str, Any], now: datetime) -> dict[str, Any]:
    cutoff = now - timedelta(days=RETENTION_DAYS)
    pruned_root: dict[str, Any] = {
        AI_MODEL_KEY: log_root[AI_MODEL_KEY],
        SYSTEM_PROMPT_KEY: log_root[SYSTEM_PROMPT_KEY],
    }

    for key, value in log_root.items():
        if key in {AI_MODEL_KEY, SYSTEM_PROMPT_KEY}:
            continue

        timestamp = _normalize_timestamp(key)
        if timestamp is not None and timestamp < cutoff:
            continue

        pruned_root[key] = value

    return pruned_root


def _update_yaml_log(
    log_file: str | Path,
    model_name: str,
    system_prompt: str,
    *,
    now: datetime,
    interaction: dict[str, Any] | None = None,
) -> None:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing_data = yaml.safe_load(handle.read()) or {}
        log_root = _prune_expired_entries(_load_log_root(existing_data, model_name, system_prompt), now)
        if interaction is not None:
            log_root[now.isoformat()] = interaction

        handle.seek(0)
        handle.truncate()
        yaml.safe_dump({LOG_ROOT_TITLE: log_root}, handle, sort_keys=False, allow_unicode=True)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def initialize_yaml_log(
    model_name: str,
    system_prompt: str,
    log_file: str | Path = OUTPUT_LOG_FILE,
    *,
    now: datetime | None = None,
) -> None:
    current_time = now or _utc_now()
    _update_yaml_log(log_file, model_name, system_prompt, now=current_time)


def log_interaction(
    model_name: str,
    system_prompt: str,
    question: str,
    response: str,
    docs_and_scores: list[tuple[Any, float]],
    log_file: str | Path = OUTPUT_LOG_FILE,
    *,
    now: datetime | None = None,
) -> None:
    current_time = now or _utc_now()
    interaction = {
        "query": question,
        "response": response,
        RETRIEVED_FILES_KEY: [
            {
                "source": doc.metadata.get("source", "unknown"),
                "similarity_score": float(score),
            }
            for doc, score in docs_and_scores
        ],
    }
    _update_yaml_log(log_file, model_name, system_prompt, now=current_time, interaction=interaction)

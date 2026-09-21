from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from shared.logger import (
    AI_MODEL_KEY,
    LOG_ROOT_TITLE,
    RETRIEVED_FILES_KEY,
    SYSTEM_PROMPT_KEY,
    configure_yaml_logging,
    initialize_yaml_log,
    log_interaction,
)


class _FakeDocument:
    def __init__(self, source: str):
        self.metadata = {"source": source}


class SharedLoggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = configure_yaml_logging(Path(self.temp_dir.name) / "MSD_Chatbot_Log.yaml")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _read_log(self) -> dict:
        with self.log_path.open("r", encoding="utf-8") as log_file:
            return yaml.safe_load(log_file)

    def test_log_structure_persists_across_reinitialization(self) -> None:
        first_time = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
        second_time = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)

        initialize_yaml_log("model-a", "system-a", self.log_path, now=first_time)
        log_interaction(
            "model-a",
            "system-a",
            "first question",
            "first answer",
            [(_FakeDocument("file-a.md"), 0.25)],
            self.log_path,
            now=first_time,
        )

        initialize_yaml_log("model-b", "system-b", self.log_path, now=second_time)
        log_interaction(
            "model-b",
            "system-b",
            "second question",
            "second answer",
            [(_FakeDocument("file-b.md"), 0.5)],
            self.log_path,
            now=second_time,
        )

        data = self._read_log()
        root = data[LOG_ROOT_TITLE]

        self.assertEqual(list(root)[:2], [AI_MODEL_KEY, SYSTEM_PROMPT_KEY])
        self.assertEqual(root[AI_MODEL_KEY], "model-b")
        self.assertEqual(root[SYSTEM_PROMPT_KEY], "system-b")
        self.assertEqual(root[first_time.isoformat()]["query"], "first question")
        self.assertEqual(root[second_time.isoformat()]["response"], "second answer")
        self.assertEqual(
            root[second_time.isoformat()][RETRIEVED_FILES_KEY],
            [{"source": "file-b.md", "similarity_score": 0.5}],
        )

    def test_prunes_only_entries_strictly_older_than_thirty_days(self) -> None:
        current_time = datetime(2026, 9, 21, 17, 5, tzinfo=timezone.utc)
        too_old = current_time - timedelta(days=30, seconds=1)
        at_cutoff = current_time - timedelta(days=30)
        recent = current_time - timedelta(days=1)

        with self.log_path.open("w", encoding="utf-8") as log_file:
            yaml.safe_dump(
                {
                    LOG_ROOT_TITLE: {
                        AI_MODEL_KEY: "stale-model",
                        SYSTEM_PROMPT_KEY: "stale-system",
                        too_old.isoformat(): {
                            "query": "old question",
                            "response": "old answer",
                            RETRIEVED_FILES_KEY: [],
                        },
                        at_cutoff.isoformat(): {
                            "query": "cutoff question",
                            "response": "cutoff answer",
                            RETRIEVED_FILES_KEY: [],
                        },
                        recent.isoformat(): {
                            "query": "recent question",
                            "response": "recent answer",
                            RETRIEVED_FILES_KEY: [],
                        },
                    }
                },
                log_file,
                sort_keys=False,
                allow_unicode=True,
            )

        initialize_yaml_log("model-a", "system-a", self.log_path, now=current_time)

        data = self._read_log()
        root = data[LOG_ROOT_TITLE]

        self.assertNotIn(too_old.isoformat(), root)
        self.assertIn(at_cutoff.isoformat(), root)
        self.assertIn(recent.isoformat(), root)


if __name__ == "__main__":
    unittest.main()

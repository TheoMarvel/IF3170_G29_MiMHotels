from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T")


class ModelSerializer: # Class for saving and loading models and preprocessing pipelines.
    
    FORMAT_VERSION: str = "1.0"

    @classmethod
    def save(cls, obj: Any, filepath: str | Path) -> Path:
        path = Path(filepath).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "format_version": cls.FORMAT_VERSION,
            "class_name": obj.__class__.__name__,
            "module_name": obj.__class__.__module__,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "instance": obj,
        }

        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        return path

    @classmethod
    def load(cls, filepath: str | Path, expected_type: Type[T] | None = None) -> T:
        # Class for loading and deserializing a pickled object from disk.
        path = Path(filepath).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")

        with path.open("rb") as f:
            payload = pickle.load(f)

        # Support both metadata-wrapped payloads and raw pickled objects
        instance = payload["instance"] if isinstance(payload, dict) and "instance" in payload else payload

        if expected_type is not None and not isinstance(instance, expected_type):
            raise TypeError(
                f"Expected loaded object to be of type '{expected_type.__name__}', "
                f"but got '{type(instance).__name__}'."
            )

        return instance
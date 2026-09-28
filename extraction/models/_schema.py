"""``$ref``-inlining for JSON schemas (llama.cpp grammar decoding ignores ``$ref``)."""

from __future__ import annotations

import copy
from typing import Any


_MAX_REF_DEPTH = 4


def inline_schema(model_cls) -> dict:
    """Return the model's JSON schema with all ``$ref``/``$defs`` inlined.

    Recursive refs are expanded at most ``_MAX_REF_DEPTH`` times per path, then become ``null``.
    """
    schema = model_cls.model_json_schema()
    defs = schema.get("$defs", {})

    def resolve(obj: Any, visiting: tuple[str, ...] = ()) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                if visiting.count(ref_name) >= _MAX_REF_DEPTH:
                    return {"type": "null"}
                return resolve(copy.deepcopy(defs[ref_name]), visiting + (ref_name,))
            return {k: resolve(v, visiting) for k, v in obj.items() if k != "$defs"}
        if isinstance(obj, list):
            return [resolve(i, visiting) for i in obj]
        return obj

    return resolve(copy.deepcopy(schema))

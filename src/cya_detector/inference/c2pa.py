"""Stage 0 C2PA verified AI-generation claim check.

Returns ``True`` only for a verified, active claim explicitly containing a
``c2pa.created`` action whose ``digitalSourceType`` is
``trainedAlgorithmicMedia``. Every other outcome — missing dependency, no
manifest, an untrusted/invalid signature, a malformed claim, an
authenticity-only claim, or any parser failure — returns ``False`` and falls
through to the real predictor. This function never returns an authenticity
verdict, never performs recursive provenance analysis, and never fetches
anything over the network (remote manifest and OCSP fetching are both
explicitly disabled below).

This is the one place in the inference pipeline that deliberately catches
every exception internally: a C2PA parsing failure is a documented *safe*
outcome (fall through to the real predictor), unlike an uncatalogued image
loading failure elsewhere in the pipeline, which is not safe to paper over.
"""

from __future__ import annotations

from pathlib import Path

_TRUSTED_VALIDATION_STATES = frozenset({"Trusted"})
_CREATED_ACTION = "c2pa.created"
_TRAINED_ALGORITHMIC_MEDIA = "trainedAlgorithmicMedia"


def has_verified_ai_generation_claim(path: Path) -> bool:
    try:
        from c2pa import Context, Reader
    except ImportError:
        return False

    try:
        context = Context.from_dict(
            {"verify": {"remote_manifest_fetch": False, "ocsp_fetch": False}}
        )
        reader = Reader.try_create(str(path), context=context)
        if reader is None:
            return False
        with reader:
            validation_state = reader.get_validation_state()
            if validation_state not in _TRUSTED_VALIDATION_STATES:
                return False
            manifest = reader.get_active_manifest()
            if not isinstance(manifest, dict):
                return False
            return _has_created_trained_algorithmic_media_action(manifest)
    except Exception:  # noqa: BLE001 — a parser failure is a documented safe False, by contract
        return False


def _has_created_trained_algorithmic_media_action(manifest: dict) -> bool:
    assertions = manifest.get("assertions")
    if not isinstance(assertions, list):
        return False
    for assertion in assertions:
        if not isinstance(assertion, dict):
            continue
        label = assertion.get("label", "")
        if not isinstance(label, str) or not label.startswith("c2pa.actions"):
            continue
        data = assertion.get("data")
        if not isinstance(data, dict):
            continue
        actions = data.get("actions")
        if not isinstance(actions, list):
            continue
        for action in actions:
            if not isinstance(action, dict):
                continue
            if action.get("action") != _CREATED_ACTION:
                continue
            digital_source_type = str(action.get("digitalSourceType", ""))
            if _TRAINED_ALGORITHMIC_MEDIA in digital_source_type:
                return True
    return False

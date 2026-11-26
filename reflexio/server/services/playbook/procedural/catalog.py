"""Address only original native-visible pieces; never bridge omitted source text."""

from typing import Any

from .models import ContractError, Input, Quotation


def catalog(
    data: Input, visible_segments: dict[int, list[str]]
) -> tuple[dict[str, Any], dict[str, Quotation], dict[str, tuple[int, str]]]:
    """Address actual visible pieces, never the artificial join between head and tail."""
    if set(visible_segments) != {s.interaction_id for s in data.sources}:
        raise ContractError(
            "Source selection requires the actual visible pieces of every source"
        )
    body = data.model_dump(mode="json")
    quotes, segments = {}, {}
    has_field_context = False
    for index, source in enumerate(body["sources"], 1):
        pieces = visible_segments[source["interaction_id"]]
        fields = data.sources[index - 1].piece_fields
        if fields is not None and len(fields) != len(pieces):
            raise ContractError("Source field labels differ from visible source pieces")
        if not pieces or any(not isinstance(p, str) or not p for p in pieces):
            raise ContractError("Visible source pieces must be nonempty text")
        if "\n\n".join(pieces) != source.pop("text"):
            raise ContractError("Visible pieces differ from the owned native source")
        source["segments"] = []
        for piece_number, text in enumerate(pieces, 1):
            segment_id = f"SEGMENT_{index}_{piece_number}"
            segments[segment_id] = (source["interaction_id"], text)
            excerpts = []
            for start in range(0, len(text), 600):
                part = text[start : start + 600]
                identity = f"EXCERPT_{index}_{piece_number}_{start // 600 + 1}"
                excerpts.append({"id": identity, "text": part})
                quotes[identity] = Quotation(
                    interaction_id=source["interaction_id"], text=part
                )
            segment: dict[str, Any] = {"id": segment_id, "excerpts": excerpts}
            if fields is not None and fields[piece_number - 1] != "content":
                segment["field"] = fields[piece_number - 1]
                has_field_context = True
            source["segments"].append(segment)
    if has_field_context:
        body["source_field_context"] = (
            "Unlabelled segments contain the original turn content. Labelled segments "
            "retain their named native source field; native_*_display is the existing "
            "renderer's representation of that metadata, not additional conversation "
            "text. Interpret each field with the containing turn's role, chronology "
            "and supplied policy. A shadow response, UI action or tool record is "
            "distinct from a direct utterance."
        )
    return body, quotes, segments

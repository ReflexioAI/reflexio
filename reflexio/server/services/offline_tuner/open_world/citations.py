"""Span-scoped evidence citation verification, tolerant of RENDERING noise only.

Why a citation resolves on its QUOTE and not only on its span ID
----------------------------------------------------------------

A span ID is a 129-character composite, ``<64 hex>:<64 hex>``. Asking a model
to transcribe one verbatim is asking it to do the single thing models are worst
at, and the observed failure is exactly that: the evidence text quoted
character-perfectly and the identifier mistyped.

Under ID-only verification that mistyped character was fatal. The caller
(:func:`..models._verified_discovery_envelope` and its held-out and
confirmation siblings) catches the rejection and rewrites the memo to
``abstain``, so a correct, well-cited hypothesis was reported as the analyst
having *chosen* to abstain -- indistinguishable downstream from a real
judgement.

What the observed failures look like
------------------------------------

Resolving on an exact quote fixes the mistyped-ID case and nothing else,
because a model also perturbs the quote. A sound ``grounded_hypothesis`` can
lose every citation to four shapes of defect:

======  ==========================================================
 shape   what differs between the model's quote and the source
======  ==========================================================
 1       the model silently corrected the user's spelling (source
         ``recieve``, model ``receive``) -- an ALTERATION of user
         text, not a rendering difference
 2       the quote is a verbatim exact substring; only the span ID
         is malformed (a hex digit dropped)
 3       a typographic apostrophe: source ``She’s`` (U+2019), model
         ``She's`` (U+0027)
 4       the same, twice, in a longer quote
======  ==========================================================

Two conclusions drive the whole design, and the second is the one worth
stating loudly:

* Shapes 2, 3 and 4 are recoverable **exactly**. Shape 2 needs quote-based ID
  resolution; 3 and 4 need one thing -- folding typographic punctuation to
  ASCII -- after which they are exact substrings. No approximate matching is
  involved anywhere.
* Shape 1 **must stay rejected**, and it is why this module deliberately does
  NOT do edit-distance matching. Its cheapest defect is a ONE-character
  substitution. Any tolerance loose enough to admit a single character edit
  would admit a model silently rewriting a user's words -- the exact failure
  verification exists to catch. Edit distance cannot separate "typed a curly
  quote as straight" from "corrected the user's spelling", because to a
  distance metric they are the same size. Folding can, because it separates
  them by KIND.

So the tolerance here is not a threshold at all. It is **edit distance zero**
against a normalised form, where normalisation is two closed, enumerated
tables: the ``_FOLD`` punctuation table, and Unicode NFKC compatibility
decomposition, which every character outside ``_FOLD`` passes through. Neither
is a metric and neither is open-ended, but the second is materially wider than
``_FOLD`` alone -- see ``_fold`` for exactly what it admits.

The resolution ladder
---------------------

Each rung is tried only when the one above it fails, and the first two are
byte-for-byte what ID-only and exact-quote verification already did:

1. the named span, exact substring, first occurrence within it;
2. exact substring, held by exactly one allowed span;
3. NORMALISED exact substring, held by exactly one allowed span.

"Unique" throughout means unique ACROSS SPANS. A quote repeated inside one span
resolves to its first occurrence rather than being refused -- every occurrence
denotes byte-identical caller text with identical provenance, so there is
nothing to choose between them. ``_locate_first`` carries the full argument.

Why this cannot be used to forge a citation
-------------------------------------------

Three independent properties:

* **The model is never credited with its own text.** ``exact_quote`` on the
  returned :class:`..models.VerifiedCitation` is
  ``span.immutable_text[start_offset:end_offset]`` -- the CALLER's bytes,
  sliced at the resolved offsets. The model's string is used only to *locate*.
  So a repaired citation cannot introduce one character the server did not
  already hold. (This is also the invariant the packaged qualification suite
  already enforces, at ``qualification.py``'s
  ``matches[0].immutable_text[start:end] != citation.exact_quote`` check.)
* **The tolerance is two fixed tables, not a metric.** An accepted quote must
  fold to an exact substring of caller text. Position by position, the model's
  character must fold to the same output as the source's character, where
  folding is ``_FOLD`` (a punctuation table, at most 5 interchangeable
  renderings per mark) for the characters it names, and per-character
  ``unicodedata.normalize("NFKC", ...)`` for **every other character**.

  NFKC is the wider of the two and its reach is easy to under-read. It is a
  closed table fixed by the Unicode standard, but it does map characters onto
  *letters, digits and letter sequences*: ``ſ`` (U+017F LATIN SMALL LETTER
  LONG S) -> ``s``, ``ﬁ`` -> ``fi``, ``①`` -> ``1``, ``²`` -> ``2``, ``℡`` ->
  ``TEL``, ``Ⅻ`` -> ``XII``, ``½`` -> ``1⁄2``, ``㈱`` -> ``(株)``. So a model
  MAY write a compatibility spelling of a source character and be accepted --
  the families are: compatibility letter forms, ligatures, sub/superscripts,
  circled and parenthesised forms, fullwidth and halfwidth forms, Roman
  numerals, letterlike symbols, and vulgar fractions.

  What that does NOT open is any freedom to change the text, for two reasons.
  First, NFKC is a *fixed* table: it maps a specific enumerated set of
  compatibility characters onto their canonical spellings and nothing else, so
  it can never turn one ordinary letter into a different ordinary letter --
  a spelling correction such as ``recieve`` -> ``receive`` is still rejected,
  and no insertion or deletion is permitted anywhere. Second, and decisively,
  the model is never credited with its bytes (see the bullet above): matching a source ``²`` by
  writing ``2`` yields a citation whose ``exact_quote`` is the source's ``²``.
  The real invariant is therefore not "the model's characters are constrained"
  but **"only the server's characters are ever attributed"**; the fold tables
  govern which server text a model is able to POINT at, never what it is able
  to say the server wrote.
* **Resolution names exactly one SPAN or it is refused.** Span uniqueness is
  checked GLOBALLY across the allowed set at every rung, never per span, because
  a quote that resolves to two spans is not a pointer and must not hand the
  model an attribution it did not earn -- two spans differ in speaker, user,
  session and trajectory. Uniqueness is NOT checked per offset within a span:
  two occurrences of one quote in one span denote the same caller bytes with
  the same provenance, so the first is taken. ``_locate_first`` states why that
  does not weaken anything.

A memo that genuinely cannot be grounded in the supplied evidence still fails
here, and still abstains. That is the property being preserved; only clerical
RENDERING of an identifier or a punctuation mark stopped being fatal.

Failure categories are deliberately unchanged, and report the PRIMARY defect
rather than a later rung's: an unknown span ID whose quote is nowhere is still
``unavailable_span``, and a known span ID whose quote is nowhere is still
``quote_absent``. The packaged qualification suite pins ``unavailable_span``
for its foreign-session citation case, and that case is unaffected -- its quote
is genuinely absent from the held-out spans the analyst was allowed to cite,
under folding as much as without it.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)

from .models import (
    CitationRepairRequest,
    EvidenceCitation,
    EvidenceSpan,
    RejectedQuote,
    VerifiedCitation,
)

type CitationVerificationCategory = Literal[
    "unavailable_span",
    "quote_absent",
    "duplicate_citation",
    "ambiguous_quote",
    # NOT a per-citation verdict. Every category above rejects ONE citation and
    # can be named to the analyst as a rejected quote; this one is a property of
    # the MEMO -- that none of its (individually valid) citations rests on a
    # turn the agent spoke. `classify_citations` therefore never produces it,
    # which is why it has no `_REPAIR_REASONS` entry: there is no single quote
    # to hand back and ask for a better transcription of.
    "no_agent_side_citation",
]


# The message each "the quote is nowhere" rejection carries. Both strings are
# the ones ID-only verification already raised, kept verbatim: callers match on
# them (`test_role_and_session_allowlists_exclude_other_frozen_documents` pins
# `match="unavailable"`), and rung 3 did not change what these two categories
# MEAN -- only that a normalised quote is now tried before either is raised.
_ABSENT_MESSAGES: dict[str, str] = {
    "unavailable_span": "citation references an unavailable span",
    "quote_absent": "citation quote is absent from its named span",
}


# The complete set of characters this module treats as interchangeable, and
# nothing else is. Each group is one punctuation mark with several Unicode
# renderings that a model substitutes for one another because they LOOK the
# same in prose; folding them changes no word and no meaning.
#
# Verified, not assumed: NFKC does NOT perform any of these mappings --
# `unicodedata.normalize("NFKC", "’") == "’"` -- which is why this
# table has to exist. U+2026 is absent because NFKC *does* expand it to "...".
#
# U+2019 -> U+0027 is the observed defect: it is the only difference in
# shapes 3 and 4 (see the module docstring). The rest of each family is
# included because it is the identical mechanism on the identical glyph.
#
# Deliberately EXCLUDED, having been considered:
#   U+0060 GRAVE ACCENT and U+00B4 ACUTE ACCENT -- not quotation marks; the
#     backtick delimits code, where folding it would erase a real distinction.
#   U+2032 PRIME / U+2033 DOUBLE PRIME -- units (minutes, feet), not quotes.
#   U+2212 MINUS SIGN -- a mathematical operator, not a hyphen.
# Folding those would widen the collision surface for defects nobody has seen.
_FOLD: dict[str, str] = {
    # Single quotation marks -> APOSTROPHE
    "‘": "'",  # LEFT SINGLE QUOTATION MARK
    "’": "'",  # RIGHT SINGLE QUOTATION MARK  <- the observed defect
    "‚": "'",  # SINGLE LOW-9 QUOTATION MARK
    "‛": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    # Double quotation marks -> QUOTATION MARK
    "“": '"',  # LEFT DOUBLE QUOTATION MARK
    "”": '"',  # RIGHT DOUBLE QUOTATION MARK
    "„": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "‟": '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    # Dashes -> HYPHEN-MINUS
    "‐": "-",  # HYPHEN
    "‑": "-",  # NON-BREAKING HYPHEN
    "‒": "-",  # FIGURE DASH
    "–": "-",  # EN DASH
    "—": "-",  # EM DASH
    "―": "-",  # HORIZONTAL BAR
}


class CitationVerificationError(ValueError):
    """A model citation failed deterministic verification."""

    category: CitationVerificationCategory

    def __init__(self, category: CitationVerificationCategory, message: str) -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class _FoldedText:
    """A normalised view of some text that can be mapped back onto it.

    ``text`` is the normalised form searched by rung 3. ``starts`` and ``ends``
    are parallel to it: ``starts[i]`` and ``ends[i]`` are the half-open RAW
    offsets of the source characters that produced ``text[i]``. A normalised
    window ``[i:j)`` therefore denotes the raw slice
    ``[starts[i], ends[j - 1])``, which is what a resolved citation is finally
    credited with. Keeping this map is what lets normalisation change lengths
    (U+2026 -> ``...``, whitespace runs -> one space) while the offsets still
    address the untouched span text.
    """

    text: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]


def _fold(text: str) -> _FoldedText:
    """Normalise text for comparison, keeping raw offsets recoverable.

    What this COVERS:

    * per-character Unicode NFKC, applied to every character ``_FOLD`` does not
      name. This is the WIDEST part of the tolerance and it is routinely
      under-read: NFKC substitutes letters, digits and whole letter sequences,
      not merely spacing and punctuation. Full-width and half-width forms;
      non-breaking and other exotic spaces; ligatures (U+FB01 ``ﬁ`` -> ``fi``);
      compatibility letter forms (U+017F ``ſ`` -> ``s``); sub- and superscripts
      (``²`` -> ``2``); circled and parenthesised forms (``①`` -> ``1``,
      ``㈱`` -> ``(株)``); Roman numerals (``Ⅻ`` -> ``XII``); letterlike
      symbols (``℡`` -> ``TEL``); vulgar fractions (``½`` -> ``1⁄2``); and
      U+2026 ellipsis -> ``...``;
    * the ``_FOLD`` table -- curly quotation marks to ASCII and the dash family
      to ``-``. NFKC does none of this, and U+2019 is the defect that actually
      occurred;
    * whitespace collapse -- any run of whitespace becomes one space, and
      leading and trailing whitespace is dropped. This absorbs the newline and
      indentation differences between a model's rendering and stored text.

    What this does NOT cover, on purpose. Note carefully that the tolerance
    surface is ``_FOLD`` **plus the NFKC compatibility mappings above** -- it
    is NOT ``_FOLD`` alone, and a reader who believes it is will conclude the
    audited exclusion list below is the only thing standing between a model and
    the source text, and delete it:

    * **case** -- folding it would double the collision surface for free, and
      case is semantic here (proper nouns, emphasis);
    * **any inserted or deleted character**;
    * **ARBITRARY letter or digit substitution** -- ``recieve`` -> ``receive``
      is the model altering user text, and it is rejected. NFKC
      is a fixed table of compatibility spellings, so it can map ``ſ`` to ``s``
      but can never map ``a`` to ``u``: outside ``_FOLD`` and that table, every
      character must match exactly. This is the single most important
      non-coverage in the module;
    * **punctuation that carries meaning** -- backtick, acute accent, primes
      and the minus sign are excluded by name in ``_FOLD``;
    * **combining-sequence composition** -- this normalises character by
      character so offsets stay alignable, so a decomposed ``e`` + U+0301 and a
      precomposed U+00E9 do not compare equal.

    Args:
        text (str): Raw text, either a span's immutable text or a model quote

    Returns:
        _FoldedText: The normalised text and its map back to raw offsets
    """
    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            run_end = index
            while run_end < length and text[run_end].isspace():
                run_end += 1
            if out:
                out.append(" ")
                starts.append(index)
                ends.append(run_end)
            index = run_end
            continue
        mapped = _FOLD.get(char)
        if mapped is None:
            mapped = unicodedata.normalize("NFKC", char)
        for produced in mapped:
            out.append(produced)
            starts.append(index)
            ends.append(index + 1)
        index += 1
    while out and out[-1] == " ":
        out.pop()
        starts.pop()
        ends.pop()
    return _FoldedText(text="".join(out), starts=tuple(starts), ends=tuple(ends))


def _locate_first(span: EvidenceSpan, quote: str) -> int | None:
    """Return the FIRST offset of ``quote`` in ``span``, or None if absent.

    Why first, and not "only when it occurs once"
    ---------------------------------------------

    This function used to refuse a second occurrence, which made a phrase
    repeated inside ONE span unconditionally uncitable: the citation fell
    through every rung and was rejected as ``ambiguous_quote``. That was in
    direct tension with the discovery prompt's own instruction --

        "Prefer the SHORTEST run of text that carries the claim you are making.
        ... Quote a few words rather than a sentence"

    -- because short agent boilerplate ("Let me check that for you.") is exactly
    the text that recurs. The rule punished an analyst for following the
    instruction it was given, and it cost the whole memo, not the one quote: a
    rejected citation downgrades the memo to an abstention.

    Uniqueness is a SPAN-level property here, and deliberately not an
    OFFSET-level one:

    * **Across spans it still must hold, and does** (see
      :func:`_resolve_by_exact_quote` and :func:`_resolve_by_normalised_quote`,
      which both still refuse a second span). Two spans differ in ``span_id``,
      ``manifest_role``, ``user_id``, ``session_id``, ``trajectory_digest`` and
      ``speaker``. Choosing between them would invent provenance the model did
      not earn -- and ``verify_grounded_discovery_citations`` reads ``speaker``
      off the RESOLVED span, so a coin flip there could manufacture agent-side
      grounding out of a user turn.
    * **Within one span there is nothing to choose between.** Every occurrence
      of one quote in one span yields a byte-identical ``exact_quote`` -- the
      same slice VALUE of ``immutable_text`` -- and identical values in every
      other field a ``VerifiedCitation`` carries. The only thing that differs is
      the offset pair, and ``citation_id``, which digests it.

    So the invariant this module exists to defend -- the server never credits
    the model with text the model supplied -- is untouched: the returned offset
    still addresses the CALLER's text, and the caller still slices its own bytes
    at it.

    What WOULD threaten that invariant is an ARBITRARY or MODEL-INFLUENCED
    choice, because it would make ``citation_id`` unstable for one citation.
    First occurrence is neither. It is a total order fixed by the server's own
    text, a pure function of ``(span, quote)``, and the model contributes
    nothing to it beyond the needle it was already required to transcribe
    exactly. Verifying the same inputs twice returns the same offsets.

    One downstream consequence, and it is the right one: a memo citing the same
    repeated phrase TWICE now resolves both to the first occurrence, and
    ``verify_citations``'s resolved-identity dedup rejects the second as
    ``duplicate_citation`` rather than rejecting both as ``ambiguous_quote``.
    That is the accurate category -- the second citation adds no evidence the
    first did not already carry, and nothing downstream could have told the two
    occurrences apart in any case.

    The empty-quote guard is not decoration. ``str.find("")`` returns 0 against
    any text, so without it an empty needle would "resolve" at offset 0 of the
    first span and credit a citation with no text at all. ``EvidenceCitation``
    already rejects a blank ``exact_quote`` and rung 3 already refuses an empty
    folded needle; this keeps the third door shut too, because the cost of
    getting it wrong here is a FORGED citation rather than a failed one.

    Args:
        span (EvidenceSpan): Caller-supplied span to search
        quote (str): Exact text the model claims to be citing

    Returns:
        int | None: The first start offset, or None when the quote is empty or
            absent from the span
    """
    if not quote:
        return None
    start_offset = span.immutable_text.find(quote)
    return None if start_offset < 0 else start_offset


def _resolve_by_exact_quote(
    quote: str, allowed_spans: Sequence[EvidenceSpan]
) -> tuple[EvidenceSpan, int, int] | None:
    """Identify the one allowed span an EXACT quote can only have come from.

    Scans every allowed span rather than stopping at the first hit, because a
    quote occurring in two SPANS is not a pointer at all -- returning the first
    would credit the model with an attribution it did not earn. A quote
    occurring twice inside ONE span IS a pointer, and resolves to its first
    occurrence; see :func:`_locate_first` for why the two cases differ.

    Args:
        quote (str): Exact text the model claims to be citing
        allowed_spans (Sequence[EvidenceSpan]): Every span the caller approved

    Returns:
        tuple[EvidenceSpan, int, int] | None: Span and raw offsets when exactly
            one span holds the quote, else None

    Raises:
        CitationVerificationError: When the quote is held by more than one span
    """
    located: tuple[EvidenceSpan, int, int] | None = None
    for span in allowed_spans:
        offset = _locate_first(span, quote)
        if offset is None:
            continue
        if located is not None:
            raise CitationVerificationError(
                "ambiguous_quote",
                "citation quote does not identify exactly one allowed span",
            )
        located = (span, offset, offset + len(quote))
    return located


def _resolve_by_normalised_quote(
    quote: str, allowed_spans: Sequence[EvidenceSpan]
) -> tuple[EvidenceSpan, int, int] | None:
    """Identify the one allowed span a quote came from, ignoring rendering.

    Folds the quote and every allowed span and then requires an EXACT, unique
    substring match between the folded forms. The offsets returned address the
    raw span text, recovered through the fold's index map, so the citation is
    credited with untouched caller bytes.

    Args:
        quote (str): Text the model claims to be citing, as it transcribed it
        allowed_spans (Sequence[EvidenceSpan]): Every span the caller approved

    Returns:
        tuple[EvidenceSpan, int, int] | None: Span and RAW offsets when exactly
            one span holds the folded quote, else None

    Raises:
        CitationVerificationError: When the folded quote is held by more than
            one span

    Note:
        Uniqueness is checked ACROSS spans only, exactly as at rung 2. A folded
        quote matching one span twice resolves to its first match -- see
        :func:`_locate_first`. Two spans that differ only in RENDERING do
        collapse to the same folded text, and that case stays fatal: they are
        still two different turns.
    """
    needle = _fold(quote).text
    if not needle:
        return None
    located: tuple[EvidenceSpan, int, int] | None = None
    for span in allowed_spans:
        folded = _fold(span.immutable_text)
        start = folded.text.find(needle)
        if start < 0:
            continue
        if located is not None:
            raise CitationVerificationError(
                "ambiguous_quote",
                "citation quote does not identify exactly one allowed span",
            )
        located = (span, folded.starts[start], folded.ends[start + len(needle) - 1])
    return located


def _resolve_citation(
    citation: EvidenceCitation,
    named: EvidenceSpan | None,
    allowed_spans: Sequence[EvidenceSpan],
) -> tuple[EvidenceSpan, int, int]:
    """Walk the resolution ladder for one citation.

    Args:
        citation (EvidenceCitation): The citation as the model emitted it
        named (EvidenceSpan | None): The span its ID names, when it names one
        allowed_spans (Sequence[EvidenceSpan]): Every span the caller approved

    Returns:
        tuple[EvidenceSpan, int, int]: The resolved span and its RAW offsets

    Raises:
        CitationVerificationError: When no rung resolves the citation
    """
    quote = citation.exact_quote
    if named is not None:
        offset = _locate_first(named, quote)
        if offset is not None:
            return (named, offset, offset + len(quote))
    resolved = _resolve_by_exact_quote(quote, allowed_spans)
    if resolved is None:
        resolved = _resolve_by_normalised_quote(quote, allowed_spans)
    if resolved is not None:
        return resolved
    # Nothing located. Report the PRIMARY defect: an ID that named no allowed
    # span is `unavailable_span`, an ID that named one whose text lacks the
    # quote is `quote_absent`.
    category: CitationVerificationCategory = (
        "quote_absent" if named is not None else "unavailable_span"
    )
    raise CitationVerificationError(category, _ABSENT_MESSAGES[category])


def verify_citations(
    citations: Sequence[EvidenceCitation],
    *,
    allowed_spans: Sequence[EvidenceSpan],
) -> tuple[VerifiedCitation, ...]:
    """Verify every model citation against exactly one caller-approved span.

    A citation locates against the span it names when it can, then against an
    exact quote, then against a normalised quote. Every rung requires an EXACT
    match -- rung 3 differs only in folding interchangeable renderings of
    punctuation first. See the module docstring for why this is not a
    relaxation of verification.

    Every returned citation carries the CALLER's text -- the slice of
    ``immutable_text`` at the resolved offsets -- and never the string the
    model transcribed, so a repaired citation cannot smuggle in a character the
    server did not already hold.

    Args:
        citations (Sequence[EvidenceCitation]): Citations as the model emitted
            them, with whatever span ID and quote it transcribed
        allowed_spans (Sequence[EvidenceSpan]): Every span this analyst was
            permitted to cite

    Returns:
        tuple[VerifiedCitation, ...]: One verified citation per input, carrying
            the RESOLVED span ID and the SOURCE quote -- either may differ from
            the model's when a transcription error was repaired

    Raises:
        CitationVerificationError: When a citation cannot be resolved to
            exactly one allowed span at exactly one offset
    """
    spans_by_id: dict[str, EvidenceSpan] = {}
    for span in allowed_spans:
        if type(span) is not EvidenceSpan:
            raise TypeError("allowed_spans must contain EvidenceSpan values")
        if span.span_id in spans_by_id:
            raise ValueError("allowed spans contain duplicate span IDs")
        spans_by_id[span.span_id] = span

    verified: list[VerifiedCitation] = []
    seen: set[tuple[str, str]] = set()
    resolved_seen: set[tuple[str, str]] = set()
    for citation in citations:
        if type(citation) is not EvidenceCitation:
            raise TypeError("citations must contain EvidenceCitation values")
        key = (citation.span_id, citation.exact_quote)
        if key in seen:
            raise CitationVerificationError("duplicate_citation", "duplicate citation")
        seen.add(key)
        span, start_offset, end_offset = _resolve_citation(
            citation, spans_by_id.get(citation.span_id), allowed_spans
        )
        # The model is credited with the CALLER's bytes, never its own. This is
        # what makes a repaired match unforgeable rather than merely unlikely.
        source_quote = span.immutable_text[start_offset:end_offset]
        # Deduplicate on the RESOLVED identity as well as the emitted one, so
        # two differently-mistyped citations of one passage cannot both be
        # credited against it.
        resolved_key = (span.span_id, source_quote)
        if resolved_key in resolved_seen:
            raise CitationVerificationError("duplicate_citation", "duplicate citation")
        resolved_seen.add(resolved_key)
        citation_id = sha256(
            canonical_json_bytes(
                {
                    "schema_version": "open-world-verified-citation-v1",
                    "span_id": span.span_id,
                    "exact_quote": source_quote,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                }
            )
        ).hexdigest()
        trajectory_digest = _trajectory_digest_from_span_id(span.span_id)
        verified.append(
            VerifiedCitation(
                citation_id=citation_id,
                span_id=span.span_id,
                exact_quote=source_quote,
                start_offset=start_offset,
                end_offset=end_offset,
                manifest_role=span.manifest_role,
                user_id=span.user_id,
                session_id=span.session_id,
                trajectory_digest=trajectory_digest,
            )
        )
    return tuple(verified)


def verify_grounded_discovery_citations(
    citations: Sequence[EvidenceCitation],
    *,
    allowed_spans: Sequence[EvidenceSpan],
    decision: str,
) -> tuple[VerifiedCitation, ...]:
    """Verify citations, and enforce the AGENT-SIDE rule for a grounded memo.

    Why this exists as a wrapper rather than as prose in a prompt
    ------------------------------------------------------------

    The discovery prompt restates the agent-side rule several times: the
    contrast "must be about the AGENT's behavior", differences on the user's
    side "are context, never the finding", an ``agent`` span is "the only
    material an agent-side contrast may be built from", and the first hard
    abstain trigger fires when the spans hold "no citable difference in the
    AGENT's behavior".

    v1.3.0 REMOVED one of those restatements -- the trigger that fired when
    "the only difference the spans will carry is a user-side one" -- on the
    ground that this wrapper already enforces the same guarantee mechanically.
    That makes this function MORE load-bearing than when it was written, not
    less: one prose restatement fewer now stands between a user-only memo and
    acceptance. Prose is not enforcement, which is the whole argument below.

    Every other clause of that contract is checked mechanically -- the span
    allowlist, byte-exact quoting, resolution ambiguity, duplication -- and this
    one was not. ``VerifiedCitation`` carries ``manifest_role``, ``user_id``,
    ``session_id`` and ``trajectory_digest``, but NO ``speaker``, so a
    ``grounded_hypothesis`` whose citations all resolve to user turns verified
    cleanly and was accepted. The rule was advisory in the one place it mattered.

    ``speaker`` needs no new binding to be trustworthy here. It is already bound
    transitively through the v2 span ID -- the resolved ``span_id`` names
    exactly one span in ``allowed_spans``, and that span's ``speaker`` is
    derived server-side by ``classify_turn_speaker`` from the stored turn's
    ``role`` column, never inferred from text and never supplied by the model.
    So this resolves the speaker by lookup and no digest changes.

    Scope, deliberately narrow:

    * only ``grounded_hypothesis`` is subject to the rule -- an abstention makes
      no agent-side claim, so it has nothing to ground;
    * an EMPTY citation set is left alone. That case is already downgraded by
      ``_verified_discovery_envelope``'s own branch, and reporting it as
      ``no_agent_side_citation`` would merge two causes this split exists to
      keep apart: "cited nothing" and "cited only the customer".

    Args:
        citations (Sequence[EvidenceCitation]): Citations as the model emitted
            them
        allowed_spans (Sequence[EvidenceSpan]): Every span this analyst was
            permitted to cite
        decision (str): The memo's own decision, verbatim

    Returns:
        tuple[VerifiedCitation, ...]: Exactly what :func:`verify_citations`
            returns

    Raises:
        CitationVerificationError: Everything :func:`verify_citations` raises,
            plus ``no_agent_side_citation`` when a grounded memo cites only
            non-agent turns
        ValueError: If a verified citation resolves to a span absent from
            ``allowed_spans`` -- a SERVER defect, and deliberately not a
            ``CitationVerificationError``, so it is never laundered into an
            abstention the analyst gets blamed for
    """
    verified = verify_citations(citations, allowed_spans=allowed_spans)
    if decision != "grounded_hypothesis" or not verified:
        return verified
    spans_by_id = {span.span_id: span for span in allowed_spans}
    for citation in verified:
        span = spans_by_id.get(citation.span_id)
        if span is None:
            raise ValueError("verified citation resolved to an unknown allowed span")
        if span.speaker == "agent":
            return verified
    raise CitationVerificationError(
        "no_agent_side_citation",
        "grounded hypothesis cites no agent-side span",
    )


def _trajectory_digest_from_span_id(span_id: str) -> str:
    parts = span_id.split(":")
    if len(parts) != 2 or any(
        len(part) != 64 or any(char not in "0123456789abcdef" for char in part)
        for part in parts
    ):
        raise ValueError("allowed span ID is not a rehydrated evidence span")
    return parts[0]


@dataclass(frozen=True)
class CitationVerdict:
    """Why one citation did or did not resolve, without aborting the rest.

    ``verify_citations`` is deliberately all-or-nothing: it raises on the FIRST
    citation it cannot resolve, so it can say *that* a memo failed but not
    *which* quotes were responsible. Telling an analyst what to repair needs the
    per-citation breakdown, so this walks the same ladder and the same dedup
    rules and records a verdict per citation instead of raising.

    ``category`` is ``None`` exactly when the citation resolved.
    """

    index: int
    span_id: str
    exact_quote: str
    category: CitationVerificationCategory | None


def classify_citations(
    citations: Sequence[EvidenceCitation],
    *,
    allowed_spans: Sequence[EvidenceSpan],
) -> tuple[CitationVerdict, ...]:
    """Report a per-citation verdict using the rules ``verify_citations`` uses.

    This is a DIAGNOSTIC view, not a second verifier, and it is not a relaxation
    of anything: it neither resolves a citation that ``verify_citations`` would
    reject nor rejects one it would accept. The two agree by construction --
    every verdict is ``None`` if and only if ``verify_citations`` returns -- and
    ``test_classification_agrees_with_verification`` pins that agreement.

    Nothing here is credited to the model: the return value carries only the
    model's own strings back, plus the category that rejected them.

    Args:
        citations (Sequence[EvidenceCitation]): Citations as the model emitted
            them
        allowed_spans (Sequence[EvidenceSpan]): Every span this analyst was
            permitted to cite

    Returns:
        tuple[CitationVerdict, ...]: One verdict per input citation, in order
    """
    spans_by_id: dict[str, EvidenceSpan] = {}
    for span in allowed_spans:
        if type(span) is not EvidenceSpan:
            raise TypeError("allowed_spans must contain EvidenceSpan values")
        if span.span_id in spans_by_id:
            raise ValueError("allowed spans contain duplicate span IDs")
        spans_by_id[span.span_id] = span

    verdicts: list[CitationVerdict] = []
    seen: set[tuple[str, str]] = set()
    resolved_seen: set[tuple[str, str]] = set()
    for index, citation in enumerate(citations):
        if type(citation) is not EvidenceCitation:
            raise TypeError("citations must contain EvidenceCitation values")
        category: CitationVerificationCategory | None = None
        key = (citation.span_id, citation.exact_quote)
        if key in seen:
            category = "duplicate_citation"
        else:
            seen.add(key)
            try:
                span, start_offset, end_offset = _resolve_citation(
                    citation, spans_by_id.get(citation.span_id), allowed_spans
                )
            except CitationVerificationError as exc:
                category = exc.category
            else:
                resolved_key = (
                    span.span_id,
                    span.immutable_text[start_offset:end_offset],
                )
                if resolved_key in resolved_seen:
                    category = "duplicate_citation"
                else:
                    resolved_seen.add(resolved_key)
        verdicts.append(
            CitationVerdict(
                index=index,
                span_id=citation.span_id,
                exact_quote=citation.exact_quote,
                category=category,
            )
        )
    return tuple(verdicts)


# What each rejection category tells the analyst, in its own words. These are
# fixed strings chosen once here, never assembled from evidence, so the repair
# payload's vocabulary cannot vary with the tenant's data.
_REPAIR_REASONS: dict[CitationVerificationCategory, str] = {
    "quote_absent": (
        "this exact text does not appear in the span you cited -- you may have "
        "translated, corrected, reworded or re-punctuated it"
    ),
    "unavailable_span": (
        "this exact text does not appear anywhere in the supplied evidence, and "
        "the span ID did not identify a span you are allowed to cite"
    ),
    "ambiguous_quote": (
        "this text appears in more than one of the evidence spans, so it does "
        "not identify a single span"
    ),
    "duplicate_citation": (
        "this citation repeats a passage another citation in the same memo "
        "already covers"
    ),
}


def citation_repair_request(
    *,
    attempt: int,
    verdicts: Sequence[CitationVerdict],
) -> CitationRepairRequest | None:
    """Describe the rejected quotes so their author can repair or drop them.

    This function takes **no spans, and no evidence of any kind** -- only the
    verdicts, whose quotes are the model's own prior output. That signature is
    the guarantee, not a comment about one: there is no evidence in scope here,
    so a repair payload cannot widen what the analyst is able to see, no matter
    how the caller invokes it. ``test_repair_request_cannot_carry_evidence``
    pins the property from the outside as well.

    Args:
        attempt (int): The 1-based number of the attempt this repair will feed,
            which is always 2 or more
        verdicts (Sequence[CitationVerdict]): Every verdict from the rejected
            attempt, resolved ones included

    Returns:
        CitationRepairRequest | None: The repair to attach to the next attempt,
            or ``None`` when nothing was rejected and there is nothing to repair
    """
    rejected = tuple(
        RejectedQuote(
            exact_quote=verdict.exact_quote,
            reason=_REPAIR_REASONS[verdict.category],
        )
        for verdict in verdicts
        if verdict.category is not None
    )
    if not rejected:
        return None
    return CitationRepairRequest(attempt=attempt, rejected_quotes=rejected)

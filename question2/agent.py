from __future__ import annotations

import json
import re
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import parse_resolution

VALID_ACTIONS = {"REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2"}
TOP_K = 3

_SYSTEM = (
    "ShopAssist Tier-1 router. Pick action (REFUND|REPLACE|SEND_ARTICLE|ESCALATE_TIER2) "
    "and one kb_article id from ARTICLES. Reply ONLY compact JSON: "
    '{"action":..,"kb_article":..,"customer_reply":..}. No analysis.'
)


@dataclass
class TicketResolution:
    action: str
    kb_article: str


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


class SupportPipeline:
    """
    triage -> retrieve -> compose -> audit, but only *compose* is an LLM call.

    triage   : deterministic, derives routing keywords from order facts + ticket text
    retrieve : lexical top-k over the KB (the huge KB is never put in a prompt)
    compose  : the single LLM call, tiny prompt, compact JSON out
    audit    : deterministic schema/KB validation; one fallback call only if it fails
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        self._index: dict[str, set[str]] | None = None

    # -- node 1: triage (no LLM) -------------------------------------------------
    def _triage(self, thread: str, order: dict) -> set[str]:
        kw = _words(thread) | _words(order.get("title", ""))
        ptype = order.get("product_type", "")
        kw |= _words(ptype)
        if ptype == "digital":
            kw |= {"digital", "license"} if order.get("license_revealed") else {"digital"}
        if ptype == "subscription" or "subscription_age_days" in order:
            kw |= {"subscription", "cooling"}
        if order.get("customized") or ptype == "custom":
            kw |= {"custom", "items"}
        if order.get("damage_reported"):
            kw |= {"damage", "replacement"}
        if ptype == "physical" and not order.get("damage_reported"):
            kw |= {"returns", "days"}
        return kw

    # -- node 2: retrieve (no LLM) -----------------------------------------------
    def _retrieve(self, keywords: set[str], k: int = TOP_K) -> list[str]:
        if self._index is None:
            self._index = {aid: _words(load_article(aid)) for aid in article_ids()}
        def score(aid: str) -> tuple[int, int]:
            title = _words(aid.replace("-", " "))
            return (len(title & keywords) * 3 + len(self._index[aid] & keywords), 0)
        ranked = sorted(article_ids(), key=score, reverse=True)
        return ranked[:k]

    # -- node 3: compose (the one LLM call) ----------------------------------------
    def _compose(self, thread: str, order: dict, art_ids: list[str], label: str = "compose") -> str:
        arts = "\n".join(f"[{a}] {load_article(a).split(chr(10), 2)[-1].strip()}" for a in art_ids)
        snapshot = json.dumps(order, separators=(",", ":"))
        messages = [
            ChatMessage("system", _SYSTEM),
            ChatMessage(
                "user",
                f"ARTICLES:\n{arts}\nTHREAD:\n{thread}\nORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
            ),
        ]
        return self.llm.chat(label, messages)

    # -- node 4: audit (no LLM) ----------------------------------------------------
    @staticmethod
    def _audit(raw: str, art_ids: list[str]):
        try:
            parsed = parse_resolution(raw)
        except Exception:
            return None
        if parsed.action in VALID_ACTIONS and parsed.kb_article in set(article_ids()):
            return parsed
        return None

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)

        keywords = self._triage(thread, order)
        art_ids = self._retrieve(keywords)
        parsed = self._audit(self._compose(thread, order, art_ids), art_ids)
        if parsed is None:  # rare fallback: widen retrieval to the whole real KB
            art_ids = article_ids()
            parsed = self._audit(self._compose(thread, order, art_ids, "compose_retry"), art_ids)
            if parsed is None:
                raise ValueError(f"Could not resolve {ticket_id}")
        return TicketResolution(parsed.action, parsed.kb_article)

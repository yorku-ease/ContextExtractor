from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from utils import clean_text

ROLE_MAP = {
    "human": "user", "user": "user",
    "gpt": "assistant", "assistant": "assistant", "bot": "assistant",
}




class WildChatAdapter:


    def normalize_messages(self, row: Dict[str, Any]) -> List[Dict[str, str]]:
        raw = row.get("conversation") or row.get("conversations") or row.get("messages") or []
        output: List[Dict[str, str]] = []
        for msg in raw:
            if not isinstance(msg, dict):
                continue
            raw_role = clean_text(msg.get("role") or msg.get("from") or msg.get("speaker")).lower()
            role = ROLE_MAP.get(raw_role)
            content = clean_text(msg.get("content") or msg.get("value") or msg.get("text"))
            if role and content:
                output.append({"role": role, "content": content})
        return output
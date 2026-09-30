from typing import Any, Dict, List

from utils import clean_text


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
}


class LMSYSAdapter:

    def normalize_messages(
        self,
        row: Dict[str, Any],
    ) -> List[Dict[str, str]]:

        raw_messages = (
            row.get("conversation")
            or row.get("conversations")
            or row.get("messages")
            or []
        )

        output: List[Dict[str, str]] = []

        for message in raw_messages:
            if not isinstance(message, dict):
                continue

            raw_role = clean_text(
                message.get("role")
                or message.get("from")
                or message.get("speaker")
            ).lower()

            role = ROLE_MAP.get(raw_role)

            content = clean_text(
                message.get("content")
                or message.get("value")
                or message.get("text")
            )

            if role and content:
                output.append({
                    "role": role,
                    "content": content,
                })

        return output
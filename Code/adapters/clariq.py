from typing import Any, Dict, List

from utils import clean_text


class ClariQAdapter:

    def normalize_messages(
        self,
        row: Dict[str, Any],
    ) -> List[Dict[str, str]]:

        initial_request = clean_text(
            row.get("initial_request")
        )

        clarification_question = clean_text(
            row.get("question")
        )

        clarification_answer = clean_text(
            row.get("answer")
        )

        if not initial_request:
            return []

        if not clarification_question:
            return []

        messages: List[Dict[str, str]] = [
            {
                "role": "user",
                "content": initial_request,
            },
            {
                "role": "assistant",
                "content": self.ensure_question_mark(
                    clarification_question
                ),
            },
        ]

        if clarification_answer:
            messages.append({
                "role": "user",
                "content": clarification_answer,
            })

        for context_turn in self.normalize_context(
            row.get("conversation_context")
        ):
            messages.extend(context_turn)

        return messages

    @staticmethod
    def ensure_question_mark(text: str) -> str:
        text = clean_text(text)

        if not text:
            return ""

        if text.endswith("?"):
            return text

        return text + "?"

    @staticmethod
    def normalize_context(
        raw_context: Any,
    ) -> List[List[Dict[str, str]]]:

        if not isinstance(raw_context, list):
            return []

        output: List[List[Dict[str, str]]] = []

        for context_turn in raw_context:
            if not isinstance(context_turn, dict):
                continue

            question = clean_text(
                context_turn.get("question")
            )

            answer = clean_text(
                context_turn.get("answer")
            )

            turn: List[Dict[str, str]] = []

            if question:
                if not question.endswith("?"):
                    question += "?"

                turn.append({
                    "role": "assistant",
                    "content": question,
                })

            if answer:
                turn.append({
                    "role": "user",
                    "content": answer,
                })

            if turn:
                output.append(turn)

        return output
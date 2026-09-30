from typing import Any, Dict, List, Optional, Tuple

from utils import clean_text


class AmbigQAAdapter:
    """
    Convert an AmbigQA multiple-interpretation record into a
    synthetic clarification dialogue.

    Important:
    The clarification question is constructed by this adapter.
    It is not an observed LLM response from the source dataset.
    """

    def normalize_messages(
        self,
        row: Dict[str, Any],
    ) -> List[Dict[str, str]]:

        original_question = clean_text(
            row.get("question")
        )

        if not original_question:
            return []

        interpretations = self.extract_interpretations(
            row
        )

        if len(interpretations) < 2:
            return []

        clarification_question = (
            self.build_clarification_question(
                interpretations
            )
        )

        first_rewrite, first_answers = (
            interpretations[0]
        )

        messages: List[Dict[str, str]] = [
            {
                "role": "user",
                "content": original_question,
            },
            {
                "role": "assistant",
                "content": clarification_question,
            },
            {
                "role": "user",
                "content": first_rewrite,
            },
        ]

        if first_answers:
            messages.append({
                "role": "assistant",
                "content": self.format_answer(
                    first_answers
                ),
            })

        return messages

    @staticmethod
    def extract_interpretations(
        row: Dict[str, Any],
    ) -> List[Tuple[str, List[str]]]:

        annotations = (
            row.get("annotations")
            or {}
        )

        qa_pairs = (
            annotations.get("qaPairs")
            or []
        )

        interpretations: List[
            Tuple[str, List[str]]
        ] = []

        for qa_pair in qa_pairs:
            if not isinstance(qa_pair, dict):
                continue

            questions = (
                qa_pair.get("question")
                or []
            )

            answers = (
                qa_pair.get("answer")
                or []
            )

            for index, question in enumerate(
                questions
            ):
                rewritten_question = clean_text(
                    question
                )

                if not rewritten_question:
                    continue

                answer_values: List[str] = []

                if index < len(answers):
                    raw_answer = answers[index]

                    if isinstance(raw_answer, list):
                        answer_values = [
                            clean_text(value)
                            for value in raw_answer
                            if clean_text(value)
                        ]

                    elif raw_answer:
                        answer_values = [
                            clean_text(raw_answer)
                        ]

                interpretations.append(
                    (
                        rewritten_question,
                        answer_values,
                    )
                )

        return interpretations

    @staticmethod
    def build_clarification_question(
        interpretations: List[
            Tuple[str, List[str]]
        ],
    ) -> str:

        alternatives = [
            question
            for question, _ in interpretations[:3]
        ]

        formatted = "; or ".join(
            f'"{alternative}"'
            for alternative in alternatives
        )

        return (
            "Could you clarify which interpretation "
            f"you mean: {formatted}?"
        )

    @staticmethod
    def format_answer(
        answers: List[str],
    ) -> str:

        if len(answers) == 1:
            return answers[0]

        return "; ".join(answers)

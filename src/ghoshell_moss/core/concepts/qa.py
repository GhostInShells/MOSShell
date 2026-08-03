from abc import ABC, abstractmethod
from typing import Literal, Callable

from pydantic import BaseModel, Field
from ghoshell_moss.message import unique_id, WithAdditional
import time

__all__ = [
    'Question', 'Answer', 'QAMeta', 'QA',
    'QAManager', 'Asker', 'Watcher',
    'YES', 'NO',
]

YES = 'yes'
NO = 'no'


class QAMeta(BaseModel, WithAdditional):
    """
    shared metadata of the question and answer
    useful in transport
    """
    id: str = Field(
        default_factory=unique_id,
    )
    issuer: str = Field(
        default='',
        description="Issuer name",
    )
    namespace: str = Field(
        default='',
        description="namespace of the question",
    )
    refer_to: str | None = Field(
        default=None,
    )
    created: float = Field(
        default_factory=lambda: round(time.time(), 4),
    )

    def new_reply(self, issuer: str) -> 'QAMeta':
        return QAMeta(
            issuer=issuer,
            namespace=self.namespace,
            refer_to=self.id,
        )


class Question(BaseModel, WithAdditional):
    meta: QAMeta | None = Field(
        default=None,
        description="Question meta",
    )
    kind: Literal['input', 'confirm', 'apply', 'choose', 'select'] = Field(
        default='input',
        description="Question kind",
    )
    content: str = Field(
        description="Question content",
    )
    max_selection: int = Field(
        default=0,
        description="max option selections",
    )
    min_selection: int = Field(
        default=0,
        description="min option selections",
    )
    options: dict[str, str] = Field(
        default_factory=dict,
        description="Question options or suggestions",
    )
    default_choices: list[str] = Field(
        default_factory=list,
    )
    canceled: str = Field(
        default='',
        description="Cancel question",
    )

    def reject(self, reason: str = '') -> 'Answer':
        """reject the question"""
        return Answer(
            content=reason,
            rejected=True,
        )

    def answer(self, content: str) -> 'Answer':
        return Answer(content=content, rejected=False, choices=[])

    def approve(self, content: str = '') -> 'Answer':
        if self.kind != 'apply':
            raise ValueError(f"Question kind {self.kind} is not apply")
        return Answer(content=content, rejected=False)

    def confirm(self, result: bool, content: str = '') -> 'Answer':
        if self.kind != 'confirm':
            raise ValueError(f"Question kind {self.kind} is not confirm")
        return Answer(
            content=content,
            choices=[YES] if result else [NO],
        )

    def choose(self, choice: str, content: str = '') -> 'Answer':
        if choice not in self.options:
            raise ValueError(f"The choice {choice} is invalid")
        if self.kind in ('choose', 'select'):
            return Answer(content=content, rejected=False, choices=[choice])
        elif self.kind == 'input':
            content = content or self.options[choice]
            return Answer(content=content, rejected=False)
        else:
            raise ValueError(f"The choice {choice} is invalid")

    def select(self, *choices: str, content: str = '') -> 'Answer':
        if self.kind != 'select':
            raise ValueError(f"Question kind {self.kind} is not select")
        choices = list(choices)
        answer = Answer(content=content, rejected=False, choices=choices)
        return answer


class Answer(BaseModel, WithAdditional):
    """
    answer data.
    """
    meta: QAMeta | None = Field(
        default=None,
    )
    content: str = Field(
        description="Answer additional content",
    )
    rejected: bool = Field(
        default=False,
        description="if the answer is rejected",
    )
    choices: list[str] = Field(
        default_factory=list,
        description="Answer choices",
    )

    def match_question(self, question: Question) -> None:
        if self.rejected:
            # reject is a valid answer itself
            return None
        choices_num = len(self.choices)
        if choices_num < question.min_selection:
            raise ValueError(f"The choice {choices_num} is too small")
        elif choices_num > question.max_selection:
            raise ValueError(f"The choice {choices_num} is too large")
        for choice in self.choices:
            if choice not in question.options:
                raise ValueError(f"The choice {choice} is invalid")
        return None


class AnswerError(RuntimeError):
    ...


class QA(ABC):
    """
    question and answer
    """

    @property
    @abstractmethod
    def question(self) -> Question:
        """the question itself"""
        ...

    @abstractmethod
    def answer_meta(self) -> QAMeta:
        """get the qa meta of the answer"""
        # generate qa meta from the question
        ...

    @property
    @abstractmethod
    def answer(self) -> Answer | None:
        """current answer, None if no answer given yet"""
        # return the answer that issuer accepted
        ...

    @abstractmethod
    def done(self) -> bool:
        """the question is done, eather answered or canceled"""
        ...

    @abstractmethod
    def replied(self) -> Answer | None:
        """the question is replied, but only issuer of the question can set it done"""
        # only Issuer can accept answer to a question,
        # once an undone question is replied, the Question is locked by `replied` but not resolved
        ...

    @abstractmethod
    def cancel(self, reason: str = '') -> None:
        """only issuer who own the question can cancel it"""
        ...

    @abstractmethod
    def owned(self) -> bool:
        """if the question is owned by current issuer"""
        ...

    @abstractmethod
    def canceled(self) -> bool:
        """if the question is canceled"""
        ...

    @abstractmethod
    def reply(self, answer: Answer) -> None:
        """try to set the answer of the question"""
        # only owner can really set the answer, others lock the question, broadcast answer, and wait final result
        ...

    @abstractmethod
    def on_answer(self, callback: Callable[[Answer], None]) -> None:
        ...

    @abstractmethod
    def on_cancel(self, callback: Callable[[Question], None]) -> None:
        ...

    @abstractmethod
    async def wait(self) -> None:
        """wait until the question is done"""
        ...

    async def __aenter__(self) -> 'QA':
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            if not self.done():
                self.cancel(f'failed: {exc_val}')
        elif not self.done():
            self.cancel(f'canceled')


class Asker(ABC):
    """
    special namespaced asker
    """

    @property
    @abstractmethod
    def issuer(self) -> str:
        """identifier of issuer or responder"""
        ...

    @abstractmethod
    def undone(self) -> list[QA]:
        """undone questions from current namespace"""
        ...

    @property
    @abstractmethod
    def namespace(self) -> str:
        """namespace of the questions"""
        ...

    def issue(
            self,
            question: Question,
            namespace: str | None = None,
    ) -> QA:
        """
        issue a question
        """
        meta = QAMeta(
            issuer=self.issuer,
            namespace=namespace if namespace is not None else self.namespace,
        )
        question.meta = meta
        return self.broadcast_question(question)

    @abstractmethod
    def broadcast_question(self, question: Question) -> QA:
        """broadcast a completed question (with meta) to anyone watching the namespace"""
        ...

    def ask(self, question: str, suggestions: list[str] | None = None) -> QA:
        """ask a question with suggestions, expect raw content as answer"""
        suggestions = suggestions or []
        options = {}
        if len(suggestions) > 0:
            for idx in range(len(suggestions)):
                options[str(idx)] = suggestions[idx]

        q = Question(
            content=question,
            kind='input',
            options=options,
            default_choices=[],
            max_selection=0,
            min_selection=0,
        )
        return self.issue(q)

    def ask_choose(
            self,
            question: str,
            options: dict[str, str],
            default: str | None = None,
    ) -> QA:
        """
        ask choose from
        """
        if default is not None and default not in options:
            raise ValueError(f"default value {default} not in options {options.keys()}")
        if len(options) < 1:
            raise ValueError(f"options must contain at least one option")

        q = Question(
            content=question,
            options=options,
            max_selection=1,
            min_selection=1,
            default_choices=[default] if default else [],
            kind='choose',
        )
        return self.issue(q)

    def ask_confirm(self, content: str, yes: str, no: str, default: bool = True) -> QA:
        question = Question(
            content=content,
            options={
                'yes': yes,
                'no': no,
            },
            max_selection=1,
            default_choices=['yes' if default else 'no'],
            kind='confirm',
        )
        return self.issue(question)

    def ask_select(
            self,
            question: str,
            options: dict[str, str],
            *,
            min_select: int = 0,
            max_select: int | None = None,
            default: list[str] | None = None,
    ) -> QA:
        if len(options) < 1:
            raise ValueError(f"options must contain at least one option")
        if max_select is not None and len(options) < max_select:
            raise ValueError(f"options must contain at most {max_select} options")

        question = Question(
            content=question,
            options=options,
            max_selection=max_select or len(options),
            min_selection=min_select,
            default_choices=default if default else [],
            kind='select',
        )
        return self.issue(question)

    def ask_approval(self, content: str) -> QA:
        """request for approval"""
        question = Question(
            content=content,
            max_selection=0,
            min_selection=0,
            options=dict(),
            kind='apply',
        )
        return self.issue(question)


class Watcher(ABC):
    """
    namespaced questions watcher
    """

    @property
    @abstractmethod
    def namespace(self) -> str:
        """namespace of the questions"""
        ...

    @abstractmethod
    def questions(
            self,
            *,
            answered: bool = False,
    ) -> list[QA]:
        """list the questions"""
        ...

    @abstractmethod
    def on_question(
            self,
            callback: Callable[[QA], None],
    ) -> None:
        """register on question"""
        ...


class QAManager(ABC):
    """
    Question and Answer protocol in dispatched system
    """

    @property
    @abstractmethod
    def issuer(self) -> str:
        ...

    @abstractmethod
    def asker(self, namespace: str) -> Asker:
        ...

    @abstractmethod
    def watch(self, namespace: str) -> Watcher:
        ...

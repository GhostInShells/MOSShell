"""Unit tests for QA concept models — Question, Answer, QAMeta."""

import pytest
from ghoshell_moss.core.concepts.qa import (
    QAMeta, Question, Answer, YES, NO,
)


class TestQAMeta:
    def test_new_reply_links_back_and_copies_namespace(self):
        parent = QAMeta(
            id='qid-1',
            issuer='asker',
            namespace='safemode',
        )
        reply_meta = parent.new_reply(issuer='responder')
        assert reply_meta.issuer == 'responder'
        assert reply_meta.namespace == 'safemode'
        assert reply_meta.refer_to == 'qid-1'
        assert reply_meta.id != parent.id


class TestQuestion:

    # --- helpers ---

    @staticmethod
    def _confirm_q():
        return Question(content='approve?', kind='confirm',
                        options={'yes': 'ok', 'no': 'nah'}, max_selection=1)

    @staticmethod
    def _apply_q():
        return Question(content='approve?', kind='apply')

    @staticmethod
    def _choose_q():
        return Question(content='pick', kind='choose',
                        options={'a': 'option a', 'b': 'option b'},
                        max_selection=1, min_selection=1)

    @staticmethod
    def _select_q():
        return Question(content='pick', kind='select',
                        options={'a': 'x', 'b': 'y', 'c': 'z'},
                        max_selection=3, min_selection=1)

    @staticmethod
    def _input_q():
        return Question(content='how?', kind='input')

    # --- tests ---

    def test_answer_plain(self):
        q = self._input_q()
        a = q.answer(content='fine')
        assert a.rejected is False
        assert a.content == 'fine'
        assert a.choices == []

    def test_reject(self):
        q = self._apply_q()
        a = q.reject(reason='busy')
        assert a.rejected is True
        assert a.content == 'busy'

    def test_confirm_yes(self):
        q = self._confirm_q()
        a = q.confirm(result=True, content='sure')
        assert a.rejected is False
        assert a.choices == [YES]

    def test_confirm_no(self):
        q = self._confirm_q()
        a = q.confirm(result=False)
        assert a.choices == [NO]

    def test_confirm_wrong_kind_raises(self):
        q = self._input_q()
        with pytest.raises(ValueError, match='not confirm'):
            q.confirm(result=True)

    def test_confirm_roundtrip_match_question(self):
        """confirm(answer with choices) must pass match_question validation."""
        q = self._confirm_q()
        a = q.confirm(result=True, content='sure')
        a.match_question(q)  # should not raise

    def test_confirm_answer_rejected_skips_match(self):
        """rejected confirm answer bypasses match_question."""
        q = self._confirm_q()
        a = q.reject(reason='no')
        a.match_question(q)  # should not raise

    def test_confirm_wrong_max_selection_raises(self):
        """confirm with max_selection=0 rejects answer with choices=["yes"]."""
        q = Question(content='bad confirm', kind='confirm',
                     options={'yes': 'ok', 'no': 'nah'},
                     max_selection=0)
        a = q.confirm(result=True)
        with pytest.raises(ValueError, match='too large'):
            a.match_question(q)

    def test_approve(self):
        q = self._apply_q()
        a = q.approve(content='approved')
        assert a.rejected is False
        assert a.content == 'approved'

    def test_approve_wrong_kind_raises(self):
        q = self._input_q()
        with pytest.raises(ValueError, match='not apply'):
            q.approve()

    def test_choose(self):
        q = self._choose_q()
        a = q.choose('a')
        assert a.choices == ['a']
        assert a.rejected is False

    def test_choose_invalid_key_raises(self):
        q = self._choose_q()
        with pytest.raises(ValueError, match='invalid'):
            q.choose('z')

    def test_choose_input_returns_no_choices(self):
        q = Question(content='x', kind='input', options={'0': 'suggestion'})
        a = q.choose('0', content='my answer')
        assert a.content == 'my answer'
        assert a.choices == []

    def test_select(self):
        q = self._select_q()
        a = q.select('a', 'c', content='picked')
        assert a.choices == ['a', 'c']
        assert a.rejected is False

    def test_select_wrong_kind_raises(self):
        q = self._input_q()
        with pytest.raises(ValueError, match='not select'):
            q.select('a')


class TestAnswer:

    @staticmethod
    def _choose_q():
        return Question(content='pick', kind='choose',
                        options={'a': 'x', 'b': 'y'},
                        max_selection=1, min_selection=1)

    @staticmethod
    def _select_q():
        return Question(content='pick', kind='select',
                        options={'a': 'x', 'b': 'y'},
                        max_selection=2, min_selection=1)

    def test_match_question_passes(self):
        q = self._choose_q()
        a = Answer(content='ok', choices=['a'])
        a.match_question(q)

    def test_match_question_rejected_skips_validation(self):
        q = self._choose_q()
        a = Answer(content='no', rejected=True)
        a.match_question(q)

    def test_match_question_too_few_choices(self):
        q = self._select_q()
        a = Answer(content='', choices=[])
        with pytest.raises(ValueError, match='too small'):
            a.match_question(q)

    def test_match_question_too_many_choices(self):
        q = self._choose_q()
        a = Answer(content='', choices=['a', 'a'])
        with pytest.raises(ValueError, match='too large'):
            a.match_question(q)

    def test_match_question_invalid_choice(self):
        q = self._choose_q()
        a = Answer(content='', choices=['z'])
        with pytest.raises(ValueError, match='invalid'):
            a.match_question(q)

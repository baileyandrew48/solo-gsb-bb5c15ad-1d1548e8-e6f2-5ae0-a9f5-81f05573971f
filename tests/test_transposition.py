"""移位审计（audit_transposition）测试。"""

import pytest

from app.assembler import AssemblyError, assemble
from app.transposition import (
    MAX_TEXT_LEN,
    TranspositionNotFoundError,
    TranspositionRequirementError,
    TranspositionTieError,
    audit_transposition,
)

# text = 甲乙丙丁戊己庚辛（8 字）
TWO_SEG = ["甲乙丙丁戊己", "戊己庚辛"]
# text = 甲乙?丁戊己庚辛（第 3 字未辨认）
WILDCARD_SEG = ["甲乙?丁戊己", "戊己庚辛"]


def test_assemble_fixtures():
    assert assemble(TWO_SEG)["text"] == "甲乙丙丁戊己庚辛"
    assert assemble(WILDCARD_SEG)["text"] == "甲乙?丁戊己庚辛"


def test_unique_move_over_three_chars():
    # 把 [2,4)「丙丁」剪下，剩余「甲乙戊己庚辛」，在插入下标 5 处放回
    r = audit_transposition(TWO_SEG, "甲乙戊己庚丙丁辛")
    assert r["text"] == "甲乙丙丁戊己庚辛"
    assert r["reference"] == "甲乙戊己庚丙丁辛"
    assert r["start"] == 2
    assert r["end"] == 4
    assert r["insert"] == 5
    assert r["moved"] == "丙丁"
    assert r["sources"] == [
        [{"segment": 0, "pos": 2}],
        [{"segment": 0, "pos": 3}],
    ]


def test_sources_cover_all_contributing_segments():
    # [4,6)「戊己」为段 0、段 1 重叠贡献，双方来源都保留
    r = audit_transposition(TWO_SEG, "甲戊己乙丙丁庚辛")
    assert (r["start"], r["end"], r["insert"], r["moved"]) == (4, 6, 1, "戊己")
    assert r["sources"] == [
        [{"segment": 0, "pos": 4}, {"segment": 1, "pos": 0}],
        [{"segment": 0, "pos": 5}, {"segment": 1, "pos": 1}],
    ]


def test_shortest_length_wins_over_longer_solution():
    # 同一参照稿另有 L=3 解（[1,4)「乙丙丁」插到 4），但最短 L=2 优先
    r = audit_transposition(TWO_SEG, "甲戊己乙丙丁庚辛")
    assert r["end"] - r["start"] == 2
    assert r["moved"] == "戊己"


def test_longer_solution_used_when_no_short_one():
    # L=2 无解；唯一解为 L=3：把 [0,3)「甲乙丙」移到剩余文本下标 4 处
    r = audit_transposition(TWO_SEG, "丁戊己庚甲乙丙辛")
    assert (r["start"], r["end"], r["insert"], r["moved"]) == (0, 3, 4, "甲乙丙")


def test_adjacent_block_swap_tie_returns_two_operations():
    # 「丙丁」右移与「戊己」左移得到同一重排文本，同长度并列
    with pytest.raises(TranspositionTieError) as exc:
        audit_transposition(TWO_SEG, "甲乙戊己丙丁庚辛")
    assert exc.value.length == 2
    ops = [(o["start"], o["end"], o["insert"], o["moved"]) for o in exc.value.ties]
    assert ops == [(2, 4, 4, "丙丁"), (4, 6, 2, "戊己")]
    for op in exc.value.ties:
        assert len(op["sources"]) == 2


def test_fragment_needs_two_known_equal_chars():
    # 参照稿在片段新位置上有一个 ?：「丙丁」对「?丁」仅 1 个两侧已知相同字，
    # 不合格；唯一解为 [4,6)「戊己」插到 2（对「戊己」2 个已知相同字）
    r = audit_transposition(TWO_SEG, "甲乙戊己?丁庚辛")
    assert (r["start"], r["end"], r["insert"], r["moved"]) == (4, 6, 2, "戊己")


def test_wildcard_in_assembled_text_pairs_free():
    # 拼接稿第 3 字为 ?：参照稿对应已知字「丙」免异字配对；
    # 「?丁」对参照「?丁」只有 1 个两侧已知相同字，该操作不合格
    r = audit_transposition(WILDCARD_SEG, "甲乙戊己?丁庚辛")
    assert (r["start"], r["end"], r["insert"], r["moved"]) == (4, 6, 2, "戊己")


def test_moved_fragment_may_contain_wildcard():
    # 移位片段含 ? 时，? 与参照稿对应字免异字配对，已知相同字够 2 个即合格：
    # [2,5)「?丁戊」插到剩余文本下标 6，片段对参照「?丁戊」有 2 个已知相同字
    r = audit_transposition(["甲乙?丁戊己庚", "己庚辛壬"], "甲乙己庚辛壬?丁戊")
    assert (r["start"], r["end"], r["insert"], r["moved"]) == (2, 5, 6, "?丁戊")


def test_single_substitution_has_no_transposition():
    # 参照稿仅一处替换异字，任何移位都无法使两稿逐字吻合
    with pytest.raises(TranspositionNotFoundError):
        audit_transposition(TWO_SEG, "甲乙丙X戊己庚辛")


def test_no_known_conflict_not_diagnosed():
    # 两稿完全一致
    with pytest.raises(TranspositionRequirementError):
        audit_transposition(TWO_SEG, "甲乙丙丁戊己庚辛")
    # 差异仅来自 ? 配对，无已知字冲突
    with pytest.raises(TranspositionRequirementError):
        audit_transposition(TWO_SEG, "甲乙?丁戊己庚?")
    with pytest.raises(TranspositionRequirementError):
        audit_transposition(TWO_SEG, "?" * 8)


def test_length_mismatch_rejected():
    with pytest.raises(TranspositionRequirementError):
        audit_transposition(TWO_SEG, "甲乙丙丁戊己庚")
    with pytest.raises(TranspositionRequirementError):
        audit_transposition(TWO_SEG, "甲乙丙丁戊己庚辛壬")


def test_reference_too_long_rejected_before_assembly():
    with pytest.raises(TranspositionRequirementError):
        audit_transposition(TWO_SEG, "字" * (MAX_TEXT_LEN + 1))


def test_max_length_boundary_accepted():
    # 120 字等长两稿：全部字互不相同，移位解唯一
    base = "".join(chr(0x4E00 + i) for i in range(MAX_TEXT_LEN))
    segments = [base[:70], base[60:]]
    assert assemble(segments)["text"] == base
    reference = base[:2] + base[4:52] + base[2:4] + base[52:]
    r = audit_transposition(segments, reference)
    assert (r["start"], r["end"], r["insert"], r["moved"]) == (2, 4, 50, base[2:4])


def test_both_sides_over_max_length_rejected():
    base = "".join(chr(0x4E00 + i) for i in range(MAX_TEXT_LEN + 1))
    with pytest.raises(TranspositionRequirementError):
        audit_transposition([base[:70], base[60:]], base)


def test_assembly_gate_failure_propagates():
    with pytest.raises(AssemblyError):
        audit_transposition(["孔子曰未知", "未知生焉", "完全无关文本"], "字" * 12)


def test_text_too_short_for_any_move():
    # 拼接稿仅 2 字：剪下 2 字后剩余文本为空，只有原位，无合格操作
    with pytest.raises(TranspositionNotFoundError):
        audit_transposition(["甲乙", "甲乙"], "乙甲")

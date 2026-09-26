"""移位审计：诊断拼接稿中一段连续文字被整体移到另一位置的情形。

提交与 :func:`app.assembler.assemble` 相同的有序 segments 与一段独立参照转写
``reference``，先复用其拼接得到拼接文本与逐字来源（沿用段数/长度限制与续接
失败规则）。两稿须**等长且不超过** :data:`MAX_TEXT_LEN`（120）字，且拼接稿
与参照稿之间须**存在已知字冲突**（某位置双方皆为已知字且不同），否则不适用
本审计（:class:`TranspositionRequirementError`）。

**枚举移位操作**：枚举拼接稿的左闭右开区间 ``[s, e)``，长度
:data:`MIN_MOVED_LEN`～:data:`MAX_MOVED_LEN`（2～12）字；把该区间**剪下**，
剩余文本 ``text[:s] + text[e:]`` 保持相对顺序，再按剩余文本的**插入下标**
``ins``（0～n-L，可落在剩余文本任意两字之间或两端）把剪下片段**放回**，
得到重排文本。排除**原位**（``ins == s``，文本不变）及**重排后未改变文本**
的操作。

**合格判定**：重排文本逐字与参照稿比较，相同字或任一侧为 ``?``（未辨认字）
算吻合，**整稿逐字均须吻合**；且移位片段在其新位置上与参照稿对应字中至少
:data:`MIN_KNOWN_EQUAL`（2）个位置**两侧皆为已知字且相同**（避免全 ``?``
片段造成误报）。

**择优**：按**移位长度最短**取解。最短长度下：

- 唯一操作：返回原区间、剪下后的插入下标、移位文本及原区间逐字 sources；
- 多种操作并列：抛出 :class:`TranspositionTieError`（携带按 ``(start, end,
  insert)`` 升序的前两种操作），由接口层转为 409；
- 任何长度都无合格操作：抛出 :class:`TranspositionNotFoundError`，由接口层
  转为 422。

实现：对每个剪下区间，剩余文本与参照稿的逐位异字数用前缀和（``remaining[:i]``
对 ``reference[:i]``）与后缀和（``remaining[i:]`` 对 ``reference[i+L:]``）
预处理，插入下标 ``ins`` 处的整稿异字数为 ``pre[ins] + suf[ins] + 移位片段
对 reference[ins:ins+L] 的异字数``，O(L) 求得；整体 O(11 · n² · L)，
n ≤ 120 时开销可忽略。

本模块只供人工复核，绝不改写拼接文本。
"""

from __future__ import annotations

from .aligner import WILDCARD
from .assembler import assemble

MAX_TEXT_LEN = 120     # 两稿（拼接稿与参照稿）等长且不超过的字数
MIN_MOVED_LEN = 2      # 移位片段最短长度
MAX_MOVED_LEN = 12     # 移位片段最长长度
MIN_KNOWN_EQUAL = 2    # 移位片段至少需要的两侧已知且相同的字数


class TranspositionRequirementError(ValueError):
    """两稿未等长、超过 120 字，或不存在已知字冲突，不适用移位诊断。"""


class TranspositionNotFoundError(ValueError):
    """不存在使重排文本与参照稿逐字吻合的 2～12 字移位操作。"""


class TranspositionTieError(ValueError):
    """最短移位长度下存在多种合格操作，无法唯一确定。"""

    def __init__(self, length: int, count: int, ties: list[dict]):
        self.length = length
        self.ties = ties
        super().__init__(
            f"移位长度 {length} 的合格操作有 {count} 种并列，无法唯一确定移位方式"
        )


def _differs(a: str, b: str) -> int:
    """逐字比较的一对字符是否不吻合：相同字或任一侧为 ``?`` 记 0，否则记 1。"""
    if a == WILDCARD or b == WILDCARD or a == b:
        return 0
    return 1


def _enumerate_ops(text: str, reference: str, moved_len: int) -> list[dict]:
    """枚举指定移位长度下的全部合格操作，按 ``(start, end, insert)`` 升序返回。"""
    n = len(text)
    ops: list[dict] = []
    for s in range(0, n - moved_len + 1):
        e = s + moved_len
        moved = text[s:e]
        remaining = text[:s] + text[e:]
        m = n - moved_len  # 剩余文本长度，插入下标取 0～m

        # pre[i] = remaining[:i] 与 reference[:i] 的异字数（前缀逐位和）
        pre = [0] * (m + 1)
        for i in range(m):
            pre[i + 1] = pre[i] + _differs(remaining[i], reference[i])
        # suf[i] = remaining[i:] 与 reference[i+moved_len:] 的异字数（后缀逐位和）
        suf = [0] * (m + 1)
        for i in range(m - 1, -1, -1):
            suf[i] = suf[i + 1] + _differs(remaining[i], reference[i + moved_len])

        for ins in range(0, m + 1):
            if ins == s:
                continue  # 原位：文本不变，排除
            # 移位片段在新位置 ins 处与参照稿对应字逐字比较
            frag_diff = 0
            known_equal = 0
            for k in range(moved_len):
                a, b = moved[k], reference[ins + k]
                if a != WILDCARD and b != WILDCARD:
                    if a == b:
                        known_equal += 1
                    else:
                        frag_diff += 1
            if frag_diff or pre[ins] + suf[ins]:
                continue  # 重排文本未与参照稿逐字吻合
            if known_equal < MIN_KNOWN_EQUAL:
                continue  # 移位片段的两侧已知且相同的字不足
            # 重排后未改变文本的操作排除（存在已知字冲突时本不会发生，防御性检查）
            if remaining[:ins] + moved + remaining[ins:] == text:
                continue
            ops.append({"start": s, "end": e, "insert": ins, "moved": moved})
    return ops


def audit_transposition(segments: list[str], reference: str) -> dict:
    """诊断拼接稿中一段 2～12 字被整体移到另一位置的操作。

    返回唯一操作时的 ``{"text", "reference", "start", "end", "insert",
    "moved", "sources"}``；拼接阶段失败原样抛出
    :class:`~app.assembler.AssemblyError`，前提不满足抛出
    :class:`TranspositionRequirementError`，无合格操作抛出
    :class:`TranspositionNotFoundError`，最短长度下多种操作并列抛出
    :class:`TranspositionTieError`。
    """
    if len(reference) > MAX_TEXT_LEN:
        raise TranspositionRequirementError(
            f"参照转写长度 {len(reference)} 字，超过 {MAX_TEXT_LEN} 字上限"
        )

    result = assemble(segments)
    text, sources = result["text"], result["sources"]
    n, m = len(text), len(reference)
    if n != m or n > MAX_TEXT_LEN:
        raise TranspositionRequirementError(
            f"拼接稿 {n} 字、参照稿 {m} 字，两稿须等长且不超过 "
            f"{MAX_TEXT_LEN} 字，不适用移位诊断"
        )

    conflicts = sum(
        1
        for i in range(n)
        if text[i] != WILDCARD and reference[i] != WILDCARD and text[i] != reference[i]
    )
    if conflicts == 0:
        raise TranspositionRequirementError(
            "拼接稿与参照稿不存在已知字冲突（两稿逐字吻合），无需诊断移位"
        )

    max_len = min(MAX_MOVED_LEN, n - 1)  # 剪下后剩余文本至少留 1 字才有意义
    for moved_len in range(MIN_MOVED_LEN, max_len + 1):
        ops = _enumerate_ops(text, reference, moved_len)
        if not ops:
            continue

        def make_op(op: dict) -> dict:
            return {
                **op,
                "sources": [
                    [dict(src) for src in char_sources]
                    for char_sources in sources[op["start"] : op["end"]]
                ],
            }

        if len(ops) > 1:
            raise TranspositionTieError(
                moved_len, len(ops), [make_op(op) for op in ops[:2]]
            )
        return {"text": text, "reference": reference, **make_op(ops[0])}

    if max_len < MIN_MOVED_LEN:
        raise TranspositionNotFoundError(
            f"拼接稿仅 {n} 字，剪下 {MIN_MOVED_LEN} 字以上片段后无剩余文本可重新插入"
        )
    raise TranspositionNotFoundError(
        f"不存在使重排文本与参照稿逐字吻合的 {MIN_MOVED_LEN}～{max_len} 字移位操作"
    )

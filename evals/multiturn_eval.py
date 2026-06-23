"""멀티턴 정확성 eval — decide_context(context 결정)의 mode/feature 재사용이 의미적으로 맞는지 측정한다.

성능(품질) 관점의 핵심 안전 속성:
  - 이전 입력을 '합쳐야 할 때만' 합치고(PATCH/USE_ACTIVE), '새 맥락'이면 절대 안 합친다(CURRENT_ONLY).
  - 잘못 합치면(false carryover) 멀쩡한 설비를 옛 수치로 오진 → 안전 위험.
  - 합쳐야 하는데 놓치면(missed carryover) 불완전 진단.

이 eval은 전체 그래프 없이 decide_context를 직접 호출해 ContextDecision을 검증한다(실 LLM 호출).
실행: PYTHONUTF8=1 PYTHONPATH=. python evals/multiturn_eval.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manufacturing_agent.context.engine import decide_context
from manufacturing_agent.context.policy import extract_machine_values
from manufacturing_agent.contracts.context import DiagnosisContext

# 직전 턴에 저장된 active 진단 context(재사용 후보). 5개 feature 모두 있음.
ACTIVE = DiagnosisContext(
    id="diag-prev-1", turn_id="t-prev", user_id="u", thread_id="th",
    features={"air_temperature": 300.0, "process_temperature": 311.0,
              "rotational_speed": 1400.0, "torque": 55.0, "tool_wear": 180.0},
    failure_types=["TWF"], prediction_summary="이전 진단: 공구마모 주의(TWF), 위험 중간.",
    created_at="2026-06-23T10:00:00",
)


def _selected(msg: str, *, active=ACTIVE, recents=None, prev_pred=True, prev_sql=None, prev_ev=None) -> dict:
    """select_context가 만드는 형태를 모사한다. current_values는 실제 추출기로 채워 충실도를 높인다."""
    return {
        "current_values": dict(extract_machine_values(msg)),
        "active_context": active,
        "recent_contexts": recents or ([active] if active else []),
        "recent_turns": [],
        "previous_prediction_summary": ACTIVE.prediction_summary if prev_pred else None,
        "previous_sql_summary": prev_sql,
        "previous_evidence_summary": prev_ev,
        "injection_in_current": False,
    }


# 각 케이스: id, follow-up message, selected 구성, 기대 mode, (선택) 검증 함수.
CASES = [
    dict(id="patch_torque", msg="아까 그 조건에서 토크만 70으로 바꾸면 위험이 어떻게 달라져?",
         sel=lambda m: _selected(m), exp_mode="PATCH_ACTIVE",
         check=lambda d: d.resolved_features.get("torque") == 70.0
                         and d.resolved_features.get("tool_wear") == 180.0),  # 나머지는 재사용
    dict(id="use_active", msg="같은 조건 그대로 다시 진단해줘",
         sel=lambda m: _selected(m), exp_mode="USE_ACTIVE",
         check=lambda d: d.resolved_features.get("tool_wear") == 180.0 and not d.changed_features),
    dict(id="new_machine_no_merge",  # ★ 핵심 안전 케이스: 합치면 안 됨
         msg="이건 다른 설비야. 토크 40만 있는데 진단해줘",
         sel=lambda m: _selected(m), exp_mode="CURRENT_ONLY",
         check=lambda d: "tool_wear" not in d.resolved_features and d.resolved_features.get("torque") == 40.0),
    dict(id="refer_result", msg="방금 진단 결과만 한 줄로 다시 요약해줘",
         sel=lambda m: _selected(m), exp_mode="REFER_ACTIVE_RESULT",
         check=lambda d: not d.resolved_features),
    dict(id="patch_toolwear", msg="다른 건 그대로 두고 공구마모만 250으로 가정하면?",
         sel=lambda m: _selected(m), exp_mode="PATCH_ACTIVE",
         check=lambda d: d.resolved_features.get("tool_wear") == 250.0
                         and d.resolved_features.get("torque") == 55.0),
    dict(id="no_prior_currentonly",  # 이전 context 없음 → short-circuit, 합칠 것도 없음
         msg="토크 60, 공구마모 210으로 진단해줘",
         sel=lambda m: _selected(m, active=None, recents=[], prev_pred=False),
         exp_mode="CURRENT_ONLY", check=lambda d: d.llm_skipped is True),
    dict(id="fresh_topic_no_merge",  # 진단과 무관한 새 주제 → 합치면 안 됨
         msg="LOTO 절차가 뭔지 설명해줘",
         sel=lambda m: _selected(m), exp_mode="CURRENT_ONLY",
         check=lambda d: not d.reused_features),
    dict(id="ambiguous_more",  # 애매: "더 자세히" — REFER 또는 CURRENT (soft)
         msg="조금 더 자세히 설명해줘",
         sel=lambda m: _selected(m), exp_mode=None, soft=True, check=None),

    # ===== 적대적/경계 케이스 (깨질 만한 것) =====
    dict(id="hard_relative_change",  # 상대값 변경: 절대 patch로 표현 안 됨
         msg="공정 온도가 지금보다 5도 더 높았으면 위험이 어떻게 돼?",
         sel=lambda m: _selected(m), exp_mode="PATCH_ACTIVE",
         # 올바르려면 process_temperature=316(311+5)이어야 함. 5로 들어가면 오류.
         check=lambda d: d.resolved_features.get("process_temperature") == 316.0),
    dict(id="hard_valueless_change",  # 값 없는 변경: patch할 수치가 없음
         msg="토크를 좀 더 올리면 위험해져?",
         sel=lambda m: _selected(m), exp_mode=None, soft=True, check=None),
    dict(id="hard_exclude_feature",  # 제외 표현: "마모만 빼고" — 선택 제외 mode 없음
         msg="공구마모만 빼고 나머지 조건 그대로 다시 진단해줘",
         sel=lambda m: _selected(m), exp_mode=None, soft=True, check=None),
    dict(id="hard_remeasure_trap",  # 재측정 함정: 같은 설비지만 새 측정 → 옛 air/temp 재사용은 위험
         msg="방금 그 설비 다시 쟀더니 토크 48, 공구마모 240 나왔어. 다시 진단해줘",
         sel=lambda m: _selected(m), exp_mode="CURRENT_ONLY",
         check=lambda d: "air_temperature" not in d.resolved_features),
    dict(id="hard_select_history",  # 다중 context 중 속성으로 지목
         msg="아까 둘 중 더 위험했던 조건에서 회전속도만 1600으로 바꾸면?",
         sel=lambda m: _selected(m, recents=[
             ACTIVE,
             DiagnosisContext(id="diag-prev-2", turn_id="t-prev2", user_id="u", thread_id="th",
                              features={"air_temperature": 302.0, "process_temperature": 318.0,
                                        "rotational_speed": 1300.0, "torque": 82.0, "tool_wear": 120.0},
                              failure_types=["OSF"], prediction_summary="이전 진단2: 과부하(OSF) 위험 높음.",
                              created_at="2026-06-23T10:05:00"),
         ]), exp_mode="SELECT_HISTORY",
         check=lambda d: d.resolved_features.get("rotational_speed") == 1600.0),
]


def main() -> int:
    rows, passed, scored = [], 0, 0
    for c in CASES:
        soft = c.get("soft", False)
        try:
            d = decide_context(c["msg"], c["sel"](c["msg"]))
            mode_ok = (c["exp_mode"] is None) or (d.mode == c["exp_mode"])
            check_ok = (c["check"] is None) or bool(c["check"](d))
            ok = mode_ok and check_ok
        except Exception as e:
            d, ok, mode_ok, check_ok = None, False, False, False
            err = f"{type(e).__name__}: {e}"
        if not soft:
            scored += 1; passed += int(ok)
        tag = "✓" if ok else ("~" if soft else "✗")
        if d is not None:
            detail = f"mode={d.mode} resolved={ {k: d.resolved_features[k] for k in sorted(d.resolved_features)} } reused={sorted(d.reused_features)}"
            why = "" if ok else f"   ← 기대 mode={c['exp_mode']}, check_ok={check_ok}"
        else:
            detail, why = f"ERROR {err}", ""
        rows.append(f"{tag} {c['id']:<24}{' (soft)' if soft else '':<7} {detail}{why}")
    print("\n=== 멀티턴 정확성 eval ===")
    print("\n".join(rows))
    print(f"\n점수(soft 제외): {passed}/{scored} = {passed/scored*100:.0f}%")
    return 0 if passed == scored else 1


if __name__ == "__main__":
    raise SystemExit(main())

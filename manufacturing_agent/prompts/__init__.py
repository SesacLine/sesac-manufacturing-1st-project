"""LLM 시스템 프롬프트 모음 패키지.

각 컴포넌트의 시스템 프롬프트를 순수 문자열 상수로 분리해 관리한다.
사용처는 submodule을 직접 import 한다(예:
    from manufacturing_agent.prompts.supervisor_planner import SUPERVISOR_PLANNER_SYS
).

원칙
- 이 패키지의 모듈은 다른 manufacturing_agent 모듈을 import 하지 않는다(순환 import 방지).
- 동적 값은 호출부에서 user 메시지로 주입한다. 여기엔 정적 지시문만 둔다.
"""

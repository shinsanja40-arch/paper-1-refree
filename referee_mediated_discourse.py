#!/usr/bin/env python3
"""
Referee-Mediated Multi-Agent Discourse Framework
=================================================
완전 버그 수정 + ML 학습 데이터 생성 최적화 버전

목적: AI 재학습을 위한 고품질 대화 데이터 생성
- 논리 전개 과정 상세 기록
- 환각 탐지 및 수정 과정 추적
- 토론자 간 상호작용 완전 재현 가능

Usage:
    python referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 42
    python referee_mediated_discourse.py --experiment good_vs_evil  --debaters 4 --seed 42

Requirements:
    pip install -r requirements.txt

Environment Variables (required):
    ANTHROPIC_API_KEY   — Anthropic Claude API key
    OPENAI_API_KEY      — OpenAI GPT API key
    GOOGLE_API_KEY      — Google Gemini API key
"""

import os
import json
import time
import argparse
import logging
import re
import random
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import deque

# ── .env 파일 로드 ────────────────────────────────────────────────────
# python-dotenv를 사용하여 프로젝트 루트의 .env에서 환경변수를 로드합니다.
# load_dotenv는 이미 환경에 설정된 변수를 덮어쓰지 않습니다 (override=False).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 미설치 시 무시

# ── tiktoken (optional) ──────────────────────────────────────────────
# 정확한 토큰 카운팅을 위해 로드합니다.
# 설치되지 않은 경우 한국어 비율 기반 가중치 폴백을 사용합니다.
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

# External dependencies
try:
    from anthropic import Anthropic
    from openai import OpenAI
    import google.generativeai as genai
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run: pip install anthropic openai google-generativeai")
    exit(1)


# ============================================================================
# Per-Turn Timeout Enforcement (스레드 기반 타임아웃 래퍼)
# ============================================================================
# config.turn_timeout 값이 있었지만 실제로 적용되지 않았습니다.
# API 호출이 무한히 걸리면 프로세스가 영원히 멈춥니다.
# 스레드 기반 래퍼를 사용하여 크로스플랫폼으로 구현했습니다.
# 기존 제안의 request_options={'timeout':...} 방식은 google-generativeai
# SDK에서 공식 지원되지 않아 TypeError를 발생시켰습니다.

class TurnTimeoutError(Exception):
    """단일 에이전트 턴이 turn_timeout을 초과한 경우 발생."""
    pass


def call_with_timeout(func, timeout_seconds: int, *args, **kwargs):
    """
    Execute func(*args, **kwargs) with a hard timeout.
    백그라운드 스레드는 강제 종료되지 않지만(Python 제한),
    메인 루프는 계속 진행됩니다.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TurnTimeoutError(
                f"Turn exceeded {timeout_seconds}s timeout"
            )


# ============================================================================
# Retry — 지수 백오프 재시도 (트랜지언트 API 오류용)
# ============================================================================

MAX_RETRIES        = 3
INITIAL_BACKOFF_S  = 1.0
BACKOFF_MULTIPLIER = 2.0
MAX_BACKOFF_S      = 30.0

_TRANSIENT_KEYWORDS = (
    "rate limit", "429", "500", "503",
    "overloaded", "temporarily", "try again",
    "connection", "timeout",
)


class DebaterSkippedError(Exception):
    """재시도 후에도 복구 불가 → 해당 토론자만 스킵."""
    pass


def _is_transient_error(exc: Exception) -> bool:
    """예외 메시지가 일시적 오류 패턴과 일치하는지 판단."""
    msg = str(exc).lower()
    return any(kw in msg for kw in _TRANSIENT_KEYWORDS)


def call_with_retry(func, timeout_seconds: int, *args, **kwargs) -> Any:
    """
    call_with_timeout 위에 지수 백오프 재시도 레이어.
    TurnTimeoutError는 재시도하지 않고 즉시 상위로 전파됩니다.
    일시적 오류만 재시도하며, 영구 오류는 즉시 DebaterSkippedError로 변환됩니다.
    """
    last_exception: Optional[Exception] = None
    backoff = INITIAL_BACKOFF_S

    for attempt in range(MAX_RETRIES + 1):
        try:
            return call_with_timeout(func, timeout_seconds, *args, **kwargs)
        except TurnTimeoutError:
            raise
        except Exception as exc:
            last_exception = exc
            if not _is_transient_error(exc) or attempt == MAX_RETRIES:
                break
            wait = min(backoff, MAX_BACKOFF_S)
            logging.getLogger(__name__).warning(
                f"Transient error (attempt {attempt+1}/{MAX_RETRIES+1}): "
                f"{exc} — retrying in {wait:.1f}s"
            )
            time.sleep(wait)
            backoff *= BACKOFF_MULTIPLIER

    raise DebaterSkippedError(
        f"Agent call failed after {MAX_RETRIES + 1} attempts"
    ) from last_exception


# ============================================================================
# Configuration Classes
# ============================================================================

class AgentRole(Enum):
    DEBATER = "debater"
    REFEREE = "referee"


class Stance(Enum):
    STRONG_A   = "strong_a"
    MODERATE_A = "moderate_a"
    NEUTRAL    = "neutral"
    MODERATE_B = "moderate_b"
    STRONG_B   = "strong_b"


@dataclass
class AgentConfig:
    name: str
    role: AgentRole
    model: str
    temperature: float
    max_tokens: int
    system_prompt: str
    persona: Optional[str] = None
    stance: Optional[Stance] = None


@dataclass
class ExperimentConfig:
    experiment_id: str
    topic: str
    description: str
    max_rounds: int
    turn_timeout: int
    deadlock_threshold: int
    seed: int
    timestamp: str
    max_context_turns: int
    debaters: List[AgentConfig]
    referee: AgentConfig


@dataclass
class LogicalStep:
    """ML 학습용: 논리 전개의 단일 스텝"""
    step_number: int
    debater_name: str
    claim: str
    reasoning: str
    evidence: List[str]
    responds_to: Optional[str] = None
    logical_type: str = "assertion"


@dataclass
class Turn:
    round_number: int
    turn_number: int
    agent_role: str
    agent_name: str
    model: str
    timestamp: str
    content: str
    tokens_used: int
    latency_ms: float
    metadata: Dict[str, Any]
    logical_steps: List[LogicalStep] = field(default_factory=list)
    references_turns: List[int] = field(default_factory=list)


@dataclass
class RefereeDecision:
    round_number: int
    turn_number: int
    target_debater: str
    claim: str
    decision: str
    reasoning: str
    evidence: List[str]
    correction: Optional[str] = None


@dataclass
class HallucinationAnnotation:
    round_number: int
    turn_number: int
    target_debater: str
    sentence_id: str
    claim: str
    is_hallucination: bool
    severity: str
    evidence: List[str]
    annotator_notes: str
    correction_applied: bool = False
    correction_text: Optional[str] = None


@dataclass
class ExperimentResults:
    config: ExperimentConfig
    turns: List[Turn]
    hallucination_annotations: List[HallucinationAnnotation]
    referee_decisions: List[RefereeDecision]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]


# ============================================================================
# Text Normalization & Deadlock Detection Helpers
# ============================================================================
# ============================================================================
# Deadlock Detection Constants
# 교착상태 탐지 상수
# ============================================================================

# Jaccard 유사도 임계값
# - 0.75 (기본): 엄격한 기준, 명확한 반복만 탐지
# - 0.65 (권장): 도돌이표 환각을 더 강하게 탐지
# - 0.50: 너무 느슨함, 정상 토론도 교착으로 오판 가능
JACCARD_THRESHOLD = 0.65




def normalize_text(text: str) -> str:
    """
    유니코드 완전 지원 텍스트 정규화.
    \\w + UNICODE 플래그로 한글·일본어·중국어 등 모든 언어를 지원합니다.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_keywords(text: str) -> frozenset:
    """
    정규화된 텍스트에서 길이 2 이상의 키워드를 추출합니다.
    한국어의 경우 조사가 붙어 단어가 짧아지기 쉽으므로
    최소 길이를 2로 설정하여 "AI", "민주" 등 핵심 키워드를 포함합니다.
    frozenset으로 반환하여 해시 가능한 객체로 만듭니다.
    """
    normalized = normalize_text(text)
    words = normalized.split()
    return frozenset(w for w in words if len(w) >= 2)


# ── 한국어 문자 정규식 ──────────────────────────────────────────────
_KOREAN_CHAR_RE = re.compile(r'[\uAC00-\uD7A3\u3131-\u318E]')


def _estimate_tokens(text: str) -> int:
    """
    토큰 수 추정.
    1순위: tiktoken (cl100k_base) — 최정밀
    2순위: 단어 수 × 한국어 비율 기반 가중치 폴백
      - 영어만: 단어당 ~1.3 토큰
      - 한국어만: 단어당 ~2.0 토큰 (조사로 인한 복수 토큰)
      - 혼용: 비율에 따라 선형 보간
    """
    if not text or not text.strip():
        return 0

    if _TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass

    words = text.split()
    if not words:
        return 0

    total_chars  = sum(len(w) for w in words)
    korean_chars = len(_KOREAN_CHAR_RE.findall(text))
    korean_ratio = korean_chars / total_chars if total_chars > 0 else 0.0

    WEIGHT_ENG = 1.3
    WEIGHT_KOR = 2.0
    weight = WEIGHT_ENG + (WEIGHT_KOR - WEIGHT_ENG) * korean_ratio
    return int(len(words) * weight)


def jaccard_similarity(set_a: frozenset, set_b: frozenset) -> float:
    """두 키워드 셋의 Jaccard 유사도를 계산합니다."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union         = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ============================================================================
# Agent Implementations
# ============================================================================

class BaseAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.conversation_history: List[Dict] = []

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

    def reset(self):
        self.conversation_history = []

    def get_last_n_turns(self, n: int = 5) -> List[Dict]:
        limit = n * 2
        return (self.conversation_history[-limit:]
                if len(self.conversation_history) > limit
                else self.conversation_history)


class ClaudeAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=api_key, timeout=30.0)

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        messages = self.get_last_n_turns(5) + [{"role": "user", "content": prompt}]

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self.config.system_prompt,
                messages=messages,
                timeout=30.0
            )
            latency_ms = (time.time() - start_time) * 1000
            content = response.content[0].text

            self.conversation_history.append({"role": "user",      "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": content})
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            return {
                "content": content,
                "tokens": response.usage.input_tokens + response.usage.output_tokens,
                "latency_ms": latency_ms,
                "model": self.config.model
            }
        except Exception as e:
            raise RuntimeError(f"Claude API error: {str(e)}") from e


class GPTAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key, timeout=30.0)

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        messages = [
            {"role": "system", "content": self.config.system_prompt}
        ] + self.get_last_n_turns(5) + [
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=30.0
            )
            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content

            self.conversation_history.append({"role": "user",      "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": content})
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            return {
                "content": content,
                "tokens": response.usage.total_tokens,
                "latency_ms": latency_ms,
                "model": self.config.model
            }
        except Exception as e:
            raise RuntimeError(f"GPT API error: {str(e)}") from e


class GeminiReferee(BaseAgent):
    """
    Referee agent using Google's Gemini — stateless design.

    매 호출마다 model.generate_content()를 직접 사용합니다.
    DebateManager가 이미 매 턴마다 최근 컨텍스트를 프롬프트에 주입하므로,
    chat 세션의 내부 히스토리는 중복 컨텍스트만 만들었습니다.

    타임아웃은 call_with_timeout() 스레드 래퍼로 적용됩니다.
    """
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=config.model,
            system_instruction=config.system_prompt
        )
        self.decision_log: deque = deque(maxlen=20)

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()

        decision_log_text = self._format_decision_log()
        full_prompt = (f"{decision_log_text}\n\n{prompt}"
                       if decision_log_text else prompt)

        try:
            # Stateless 호출 — chat 세션 없음
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                )
            )
            latency_ms = (time.time() - start_time) * 1000
            content = response.text

            try:
                token_count = self.model.count_tokens(content)
                tokens = token_count.total_tokens
            except Exception:
                tokens = _estimate_tokens(content)

            return {
                "content": content,
                "tokens": tokens,
                "latency_ms": latency_ms,
                "model": self.config.model,
                "grounding_metadata": getattr(response, 'grounding_metadata', None)
            }
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}") from e

    def _format_decision_log(self) -> str:
        if not self.decision_log:
            return ""
        log_text = "[PREVIOUS DECISIONS FOR CONSISTENCY]\n"
        log_text += "Maintain consistency with your previous decisions:\n\n"
        for idx, decision in enumerate(list(self.decision_log)[-5:], 1):
            log_text += f"{idx}. Round {decision.round_number}, {decision.target_debater}: "
            log_text += f"'{decision.claim}' → {decision.decision}\n"
            log_text += f"   Reasoning: {decision.reasoning[:100]}...\n\n"
        return log_text

    def add_decision(self, decision: RefereeDecision):
        self.decision_log.append(decision)

    def reset(self):
        super().reset()
        self.decision_log.clear()


# ============================================================================
# Debate Manager
# ============================================================================

class DebateManager:
    """다중 에이전트 라운드 기반 토론 관리자"""

    def __init__(self, config: ExperimentConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.turns: List[Turn] = []
        # Jaccard 기반 교착 탐지 — per-debater 자기 반복 추적
        self.per_debater_keyword_history: Dict[str, deque] = {}
        # 그룹 교착 카운터 (모든 토론자가 동시에 반복 패턴을 보일 때 증가)
        self.consecutive_all_repeat_count: int = 0

        self.all_referee_decisions: List[RefereeDecision] = []

        self.debaters = [self._create_agent(d) for d in config.debaters]
        self.referee  = self._create_agent(config.referee)

        self.logger.info(f"Initialized {len(self.debaters)} debaters")

    def _create_agent(self, agent_config: AgentConfig) -> BaseAgent:
        model_lower = agent_config.model.lower()
        if "claude" in model_lower:
            return ClaudeAgent(agent_config)
        elif "gpt" in model_lower:
            return GPTAgent(agent_config)
        elif "gemini" in model_lower:
            return GeminiReferee(agent_config)
        else:
            raise ValueError(f"Unknown model: {agent_config.model}")

    def _create_turn(self, round_num: int, turn_num: int, agent: BaseAgent,
                     response: Dict[str, Any], role: str) -> Turn:
        return Turn(
            round_number=round_num,
            turn_number=turn_num,
            agent_role=role,
            agent_name=agent.config.name,
            model=agent.config.model,
            timestamp=datetime.now().isoformat(),
            content=response["content"],
            tokens_used=response["tokens"],
            latency_ms=response["latency_ms"],
            metadata=response
        )

    def _detect_deadlock(self, content: str, debater_name: str) -> bool:
        """
        Per-Debater 자기 반복 탐지 (Jaccard 유사도 기반).

        동일한 토론자가 자신의 이전 발언과 JACCARD_THRESHOLD 이상의
        유사도를 보이면 True를 반환합니다.

        ■ 합의 ≠ 교착
            A와 B가 같은 키워드로 합의하는 것은 교착이 아닙니다.
            자기 자신의 이전 발언과만 비교하여 오판을 방지합니다.

        ■ 그룹 교착 판정은 _check_group_deadlock()에서 별도 수행
            토론자 루프가 완료된 후에만 호출되어 정확한 타이밍을 보장합니다.
        """
        keywords = extract_keywords(content)
        if not keywords:
            return False

        window_size = self.config.deadlock_threshold
        if debater_name not in self.per_debater_keyword_history:
            self.per_debater_keyword_history[debater_name] = deque(
                maxlen=window_size
            )

        history = self.per_debater_keyword_history[debater_name]

        # 자기 자신의 이전 발언과만 비교
        self_repeat = any(
            jaccard_similarity(keywords, prev) >= JACCARD_THRESHOLD
            for prev in history
        )

        # 현재 발언을 히스토리에 추가 (체크 이후)
        history.append(keywords)

        if self_repeat:
            self.logger.warning(
                f"Self-repeat detected: {debater_name} "
                f"(threshold={JACCARD_THRESHOLD})"
            )

        return self_repeat

    def _check_group_deadlock(self) -> bool:
        """
        그룹 교착 판정 — 라운드의 모든 토론자가 발언한 후에만 호출됩니다.

        모든 토론자가 각각 자신의 직전 발언과 JACCARD_THRESHOLD 이상의
        유사도를 보이는 라운드가 deadlock_threshold회 연속으로 쌓이면
        전체 토론이 교착 상태로 판정됩니다.

        호출 타이밍:
            run_debate의 토론자 루프가 완료된 직후 (심판 호출 전)
            → 이 시점에서 모든 토론자의 현재 라운드 발언이
              per_debater_keyword_history에 포함되어 있음
        """
        num_debaters = len(self.per_debater_keyword_history)
        all_have_history = all(
            len(h) >= 2
            for h in self.per_debater_keyword_history.values()
        )

        if num_debaters < 2 or not all_have_history:
            self.consecutive_all_repeat_count = 0
            return False

        # 각 토론자의 마지막 두 발언 사이 유사도 체크
        all_repeating = all(
            jaccard_similarity(list(h)[-1], list(h)[-2]) >= JACCARD_THRESHOLD
            for h in self.per_debater_keyword_history.values()
        )

        if all_repeating:
            self.consecutive_all_repeat_count += 1
            self.logger.warning(
                f"Group deadlock signal "
                f"(count: {self.consecutive_all_repeat_count}/"
                f"{self.config.deadlock_threshold})"
            )
        else:
            self.consecutive_all_repeat_count = 0

        return self.consecutive_all_repeat_count >= self.config.deadlock_threshold

    def _get_recent_context(self) -> str:
        recent_turns = (self.turns[-self.config.max_context_turns:]
                        if self.turns else [])
        if not recent_turns:
            return ""

        context = "[RECENT DEBATE CONTEXT]\n"
        for turn in recent_turns:
            tag = f"[{turn.agent_name}]"
            content_summary = (turn.content[:200] + "..."
                               if len(turn.content) > 200
                               else turn.content)
            context += f"{tag}\n{content_summary}\n\n"
        return context

    def run_debate(self, initial_prompt: str) -> Tuple[List[Turn], List[RefereeDecision]]:
        """
        라운드 기반 토론 실행.
        각 에이전트 호출에 turn_timeout을 스레드 타임아웃으로 적용합니다.
        """
        self.logger.info(f"Starting debate: {self.config.topic}")
        turn_counter = 0

        for round_num in range(1, self.config.max_rounds + 1):
            self.logger.info("=" * 80)
            self.logger.info(f"ROUND {round_num} START")
            self.logger.info("=" * 80)

            round_statements: List[Tuple[str, str, int]] = []

            # ── 각 토론자 차례 ────────────────────────────────────────
            for debater_idx, debater in enumerate(self.debaters):
                turn_counter += 1
                debater_role = f"debater_{debater_idx + 1}"

                self.logger.info(
                    f"Round {round_num}, Turn {turn_counter}: {debater.config.name}"
                )

                prompt = self._build_debater_prompt(
                    round_num, debater_idx,
                    debater.config.name, round_statements, initial_prompt
                )

                try:
                    # 스레드 타임아웃 + 지수 백오프 재시도
                    response = call_with_retry(
                        debater.generate_response,
                        self.config.turn_timeout,
                        prompt
                    )
                    turn = self._create_turn(
                        round_num, turn_counter, debater, response, debater_role
                    )
                    turn.references_turns = self._extract_references(
                        response['content'], round_statements
                    )
                    self.turns.append(turn)
                    round_statements.append(
                        (debater.config.name, response['content'], turn_counter)
                    )

                    # per-debater 자기 반복 체크 (경고 로깅용)
                    # 그룹 교착 판정은 토론자 루프 종료 후 수행
                    self._detect_deadlock(response['content'], debater.config.name)

                except TurnTimeoutError:
                    self.logger.warning(
                        f"{debater.config.name} Turn {turn_counter} timed out "
                        f"({self.config.turn_timeout}s)"
                    )
                    round_statements.append(
                        (debater.config.name,
                         "[TIMEOUT: No response within time limit]",
                         turn_counter)
                    )
                    continue

                except DebaterSkippedError as e:
                    self.logger.warning(
                        f"{debater.config.name} Turn {turn_counter} skipped "
                        f"after retries: {e}"
                    )
                    round_statements.append(
                        (debater.config.name,
                         "[SKIPPED: API call failed after retries]",
                         turn_counter)
                    )
                    continue

                except Exception as e:
                    self.logger.error(f"Error in {debater.config.name}'s turn: {e}")
                    round_statements.append(
                        (debater.config.name,
                         "[ERROR: Failed to generate response]",
                         turn_counter)
                    )
                    continue

            # ── 라운드 종료 — 그룹 교착 체크 ──────────────────────────
            # 모든 토론자의 발언이 완료된 시점에서 그룹 교착을 판정합니다.
            if self._check_group_deadlock():
                self.logger.warning(f"Group deadlock confirmed in round {round_num}")
                return self.turns, self.all_referee_decisions

            # ── 심판 개입 ────────────────────────────────────────────
            self.logger.info(
                f"Round {round_num}: Referee reviewing {len(round_statements)} statements"
            )

            referee_prompt = self._build_referee_prompt(round_num, round_statements)

            try:
                turn_counter += 1
                # 심판 호출에도 동일한 스레드 타임아웃 + 재시도
                referee_response = call_with_retry(
                    self.referee.generate_response,
                    self.config.turn_timeout,
                    referee_prompt
                )
                turn = self._create_turn(
                    round_num, turn_counter, self.referee, referee_response, "referee"
                )
                self.turns.append(turn)
                self._extract_and_log_decisions(
                    round_num, round_statements, referee_response['content']
                )

            except TurnTimeoutError:
                self.logger.warning(
                    f"Referee Turn {turn_counter} timed out "
                    f"({self.config.turn_timeout}s)"
                )
                dummy_response = {
                    "content": json.dumps({
                        "round": round_num, "claims": [],
                        "overall_assessment": "TIMEOUT: Referee did not respond in time"
                    }),
                    "tokens": 0, "latency_ms": 0, "model": self.config.referee.model
                }
                turn = self._create_turn(
                    round_num, turn_counter, self.referee, dummy_response, "referee"
                )
                self.turns.append(turn)

            except Exception as e:
                self.logger.error(f"Error in referee review: {e}")
                dummy_response = {
                    "content": json.dumps({
                        "round": round_num, "claims": [],
                        "overall_assessment": f"ERROR: {str(e)}"
                    }),
                    "tokens": 0, "latency_ms": 0, "model": self.config.referee.model
                }
                turn = self._create_turn(
                    round_num, turn_counter, self.referee, dummy_response, "referee"
                )
                self.turns.append(turn)

        self.logger.info(f"Debate completed: {len(self.turns)} total turns")
        return self.turns, self.all_referee_decisions

    def _build_debater_prompt(self, round_num: int, debater_idx: int,
                              debater_name: str,
                              round_statements: List[Tuple[str, str, int]],
                              initial_prompt: str) -> str:
        if round_num == 1 and debater_idx == 0:
            return initial_prompt

        prompt_parts = []

        recent_context = self._get_recent_context()
        if recent_context:
            prompt_parts.append(recent_context)

        if round_num > 1:
            referee_turns = [t for t in self.turns if t.agent_role == "referee"]
            if referee_turns:
                last_referee = referee_turns[-1]
                prompt_parts.append(
                    f"[REFEREE'S LAST REVIEW]\n{last_referee.content}\n"
                )
                prompt_parts.append(
                    f"⚠️ Pay special attention to feedback about YOUR "
                    f"({debater_name}) claims.\n"
                )

        if round_statements:
            prompt_parts.append("[CURRENT ROUND - YOUR COLLEAGUES' STATEMENTS]")
            for colleague_name, colleague_statement, turn_num in round_statements:
                statement_preview = (
                    colleague_statement[:300] + "..."
                    if len(colleague_statement) > 300
                    else colleague_statement
                )
                prompt_parts.append(
                    f"\n{colleague_name} (Turn {turn_num}):\n{statement_preview}\n"
                )

        instructions = """
[YOUR TASK]
Provide your argument addressing:
1. The referee's criticisms specifically about YOUR claims
2. Your colleagues' points in this round (agree, disagree, or build upon)
3. The opposing side's arguments

IMPORTANT: Cite sources for factual claims. Engage directly with others' arguments.
"""
        prompt_parts.append(instructions)
        return "\n".join(prompt_parts)

    def _build_referee_prompt(self, round_num: int,
                              statements: List[Tuple[str, str, int]]) -> str:
        prompt = f"""[ROUND {round_num} FACT-CHECK REVIEW]

Review ALL statements. For EACH claim, specify WHO made it.

Required JSON structure:
{{
  "round": {round_num},
  "claims": [
    {{
      "debater": "<EXACT debater name>",
      "turn_number": <turn number>,
      "quote": "<exact quote>",
      "status": "<CORRECT|FACTUAL_ERROR|UNVERIFIABLE|MISLEADING>",
      "severity": "<low|medium|high>",
      "evidence": ["<source URLs>"],
      "correction": "<correction if needed>",
      "notes": "<your analysis>"
    }}
  ],
  "overall_assessment": "<summary>"
}}

Statements to review:

"""
        for idx, (debater_name, content, turn_num) in enumerate(statements, 1):
            content_for_review = (
                content[:2000] + "... [truncated]"
                if len(content) > 2000
                else content
            )
            prompt += f"\n{idx}. {debater_name} (Turn {turn_num}):\n{content_for_review}\n\n"

        prompt += "\nBe EXTREMELY CRITICAL. Specify WHO made WHICH claim."
        return prompt

    def _extract_and_log_decisions(self, round_num: int,
                                   round_statements: List[Tuple[str, str, int]],
                                   referee_content: str):
        try:
            # LLM이 ```json … ``` 형태로 응답하는 경우 마커를 먼저 제거합니다.
            cleaned = re.sub(
                r'```(?:json)?\s*', '', referee_content
            ).replace('```', '')
            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if json_match:
                parsed = json.loads(json_match.group(0))
                for claim_obj in parsed.get('claims', []):
                    if not isinstance(claim_obj, dict):
                        continue

                    decision = RefereeDecision(
                        round_number=round_num,
                        turn_number=claim_obj.get('turn_number', 0),
                        target_debater=claim_obj.get('debater', 'UNKNOWN'),
                        claim=claim_obj.get('quote', 'N/A')[:100],
                        decision=claim_obj.get('status', 'UNKNOWN'),
                        reasoning=claim_obj.get('notes', '')[:200],
                        evidence=claim_obj.get('evidence', [])[:3],
                        correction=claim_obj.get('correction', None)
                    )

                    self.all_referee_decisions.append(decision)
                    if isinstance(self.referee, GeminiReferee):
                        self.referee.add_decision(decision)

        except Exception as e:
            self.logger.warning(f"Failed to extract decisions: {e}")

    def _extract_references(self, content: str,
                            round_statements: List[Tuple[str, str, int]]) -> List[int]:
        references = []
        for name, _statement, turn_num in round_statements:
            if name in content:
                references.append(turn_num)
        return references


# ============================================================================
# Evaluation Module
# ============================================================================

class HallucinationEvaluator:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def extract_referee_annotations(self, turns: List[Turn],
                                    referee_decisions: List[RefereeDecision]
                                    ) -> List[HallucinationAnnotation]:
        annotations = []
        for decision in referee_decisions:
            is_hallucination = decision.decision in [
                'FACTUAL_ERROR', 'UNVERIFIABLE', 'MISLEADING'
            ]
            annotations.append(HallucinationAnnotation(
                round_number=decision.round_number,
                turn_number=decision.turn_number,
                target_debater=decision.target_debater,
                sentence_id=f"round_{decision.round_number}_turn_{decision.turn_number}",
                claim=decision.claim,
                is_hallucination=is_hallucination,
                severity=decision.decision.lower(),
                evidence=decision.evidence,
                annotator_notes=decision.reasoning,
                correction_applied=decision.correction is not None,
                correction_text=decision.correction
            ))
        return annotations

    def calculate_metrics(self, turns: List[Turn],
                          annotations: List[HallucinationAnnotation]
                          ) -> Dict[str, float]:
        debater_turns        = [t for t in turns if "debater" in t.agent_role]
        total_hallucinations = sum(1 for a in annotations if a.is_hallucination)
        factual_errors       = sum(1 for a in annotations if 'error'        in a.severity.lower())
        unverifiable         = sum(1 for a in annotations if 'unverifiable' in a.severity.lower())
        misleading           = sum(1 for a in annotations if 'misleading'   in a.severity.lower())
        corrected            = sum(1 for a in annotations if a.correction_applied)

        return {
            "total_turns":            len(turns),
            "debater_turns":          len(debater_turns),
            "referee_interventions":  len([t for t in turns if t.agent_role == "referee"]),
            "total_hallucinations":   total_hallucinations,
            "hallucination_rate":     (total_hallucinations / len(debater_turns)
                                       if debater_turns else 0),
            "factual_errors":         factual_errors,
            "unverifiable_claims":    unverifiable,
            "misleading_claims":      misleading,
            "corrections_provided":   corrected,
            "correction_rate":        (corrected / total_hallucinations
                                       if total_hallucinations else 0)
        }


# ============================================================================
# Experiment Runner
# ============================================================================

class ExperimentRunner:
    def __init__(self, experiment_name: str, num_debaters: int = 4,
                 seed: int = 42, output_base_dir: str = "outputs"):
        self.experiment_name = experiment_name
        self.num_debaters    = num_debaters
        self.seed            = seed
        self.timestamp       = datetime.now().isoformat()

        random.seed(seed)

        self.logger = self._setup_logger()

        self.output_dir = (
            f"{output_base_dir}/{experiment_name}_"
            f"{num_debaters}d_{self.timestamp.replace(':', '-')}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """핸들러 중복 가드 포함 로거 설정"""
        logger = logging.getLogger(
            f"{self.experiment_name}_{self.num_debaters}d_{id(self)}"
        )
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def _create_balanced_debaters(self, experiment_name: str) -> List[AgentConfig]:
        if experiment_name == "nuclear_energy":
            topic_a, topic_b = "nuclear energy", "renewable energy"
        elif experiment_name == "good_vs_evil":
            topic_a, topic_b = "human goodness", "human evil capacity"
        else:
            topic_a, topic_b = "option A", "option B"

        models = [
            "claude-3-5-sonnet-20241022", "gpt-4o",
            "claude-3-5-sonnet-20241022", "gpt-4o"
        ]

        if self.num_debaters == 4:
            stances = [Stance.STRONG_A, Stance.MODERATE_A,
                       Stance.STRONG_B, Stance.MODERATE_B]
        elif self.num_debaters == 6:
            stances = [Stance.STRONG_A, Stance.MODERATE_A, Stance.NEUTRAL,
                       Stance.STRONG_B, Stance.MODERATE_B, Stance.NEUTRAL]
        else:
            half = self.num_debaters // 2
            stances = [Stance.STRONG_A] * half + [Stance.STRONG_B] * half

        stance_templates = {
            Stance.STRONG_A:   (f"Strong {topic_a.title()} Advocate",
                                f"You strongly support {topic_a}. Engage with others' arguments."),
            Stance.MODERATE_A: (f"Moderate {topic_a.title()} Supporter",
                                f"You support {topic_a} but acknowledge merits of {topic_b}."),
            Stance.STRONG_B:   (f"Strong {topic_b.title()} Advocate",
                                f"You strongly support {topic_b}. Engage with others' arguments."),
            Stance.MODERATE_B: (f"Moderate {topic_b.title()} Supporter",
                                f"You support {topic_b} but acknowledge merits of {topic_a}."),
            Stance.NEUTRAL:    ("Neutral Analyst",
                                f"You are neutral. Evaluate both {topic_a} and {topic_b} objectively."),
        }

        debaters = []
        for i in range(self.num_debaters):
            stance = stances[i] if i < len(stances) else Stance.NEUTRAL
            model  = models[i % len(models)]
            name, prompt = stance_templates[stance]
            debaters.append(AgentConfig(
                name=name, role=AgentRole.DEBATER, model=model,
                temperature=0.7, max_tokens=1000,
                system_prompt=prompt, stance=stance
            ))
        return debaters

    def _get_experiment_config(self, experiment_name: str) -> ExperimentConfig:
        debaters = self._create_balanced_debaters(experiment_name)
        referee  = AgentConfig(
            name="Critical Fact Checker",
            role=AgentRole.REFEREE,
            model="gemini-1.5-pro",
            temperature=0.3,
            max_tokens=2000,
            system_prompt="""You are an extremely critical fact-checking referee.

CRITICAL: For EVERY claim, you MUST specify:
1. WHO made it (exact debater name)
2. WHICH turn number
3. The exact quote
4. Your verdict with evidence

Respond ONLY in valid JSON format with the structure specified."""
        )

        return ExperimentConfig(
            experiment_id=f"{experiment_name}_{self.num_debaters}d_{self.timestamp}",
            topic=experiment_name,
            description=f"ML training data: {self.num_debaters} debaters",
            max_rounds=5,
            turn_timeout=60,
            deadlock_threshold=3,
            seed=self.seed,
            timestamp=self.timestamp,
            max_context_turns=10,
            debaters=debaters,
            referee=referee
        )

    def run(self) -> ExperimentResults:
        """
        실험 파이프라인 실행.
        run_debate가 중간에 충돌하면 완료된 턴을 부분 저장합니다.
        """
        self.logger.info("=" * 80)
        self.logger.info(
            f"Experiment: {self.experiment_name}, Debaters: {self.num_debaters}"
        )
        self.logger.info("=" * 80)

        config         = self._get_experiment_config(self.experiment_name)
        debate_manager = DebateManager(config, self.logger)
        initial_prompt = f"Present your argument on: {config.topic}. Cite sources."

        # run_debate를 try-except로 감싸여 중간 충돌 시 부분 결과 저장
        try:
            turns, referee_decisions = debate_manager.run_debate(initial_prompt)
        except Exception as e:
            self.logger.error(f"Debate crashed: {type(e).__name__}: {e}")
            self.logger.warning("Saving partial results from completed turns.")
            turns             = debate_manager.turns
            referee_decisions = debate_manager.all_referee_decisions

        evaluator   = HallucinationEvaluator(self.logger)
        annotations = evaluator.extract_referee_annotations(turns, referee_decisions)
        metrics     = evaluator.calculate_metrics(turns, annotations)

        results = ExperimentResults(
            config=config, turns=turns,
            hallucination_annotations=annotations,
            referee_decisions=referee_decisions,
            metrics=metrics,
            metadata={"output_directory": self.output_dir}
        )

        self._save_results(results)

        self.logger.info("=" * 80)
        self.logger.info(f"Total turns:        {metrics['total_turns']}")
        self.logger.info(f"Hallucination rate: {metrics['hallucination_rate']:.2%}")
        self.logger.info(f"Corrections:        {metrics['corrections_provided']}")
        self.logger.info("=" * 80)

        return results

    def _save_results(self, results: ExperimentResults):
        files = {
            "full_transcript.json":           [asdict(t) for t in results.turns],
            "referee_decisions.json":         [asdict(d) for d in results.referee_decisions],
            "hallucination_annotations.json": [asdict(a) for a in results.hallucination_annotations],
            "metrics.json":                   results.metrics,
        }
        for filename, data in files.items():
            path = os.path.join(self.output_dir, filename)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(results.config), f, indent=2,
                      ensure_ascii=False, default=str)

        self.logger.info(f"Results saved to {self.output_dir}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ML Training Data Generator — Referee-Mediated Discourse"
    )
    parser.add_argument(
        "--experiment",
        choices=["nuclear_energy", "good_vs_evil"],
        required=True,
        help="Which experiment to run"
    )
    parser.add_argument(
        "--debaters",
        type=int,
        default=4,
        help="Number of debaters (must be >= 4 and even)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Base directory for output files"
    )

    args = parser.parse_args()

    if args.debaters < 4 or args.debaters % 2 != 0:
        print("ERROR: --debaters must be >= 4 and even")
        return 1

    required_keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]
    missing_keys  = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        print("\n❌ Missing API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        return 1

    try:
        runner  = ExperimentRunner(
            args.experiment, args.debaters, args.seed, args.output_dir
        )
        results = runner.run()

        print(f"\n✅ Experiment completed!")
        print(f"📁 Results: {runner.output_dir}\n")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

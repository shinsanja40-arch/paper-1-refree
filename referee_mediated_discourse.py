#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Referee-Mediated Multi-Agent Discourse Framework
# ---------------------------------------------------------------------------
# Copyright (c) 2026 Cheongwon Choi <ccw1914@naver.com>
# Licensed under CC BY-NC 4.0
#   - Personal use allowed.  Commercial use prohibited.
#   - Attribution required.
#   - Full terms: https://creativecommons.org/licenses/by-nc/4.0/
# ---------------------------------------------------------------------------
# 완전 버그 수정 + ML 학습 데이터 생성 최적화 버전  (v5.3.0)
#
# 목적: AI 재학습을 위한 고품질 대화 데이터 생성
# - 논리 전개 과정 상세 기록
# - 환각 탐지 및 수정 과정 추적
# - 토론자 간 상호작용 완전 재현 가능
#
# Usage:
#     python referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 42
#     python referee_mediated_discourse.py --experiment good_vs_evil  --debaters 4 --seed 42
#
# Requirements:
#     pip install -r requirements.txt
#
# Environment Variables (required):
#     ANTHROPIC_API_KEY   — Anthropic Claude API key
#     OPENAI_API_KEY      — OpenAI GPT API key
#     GOOGLE_API_KEY      — Google Gemini API key
# ---------------------------------------------------------------------------

import os
import json
import time
import argparse
import logging
import logging.handlers
import re
import random
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque

# ── .env 파일 로드 ────────────────────────────────────────────────────────
# python-dotenv를 사용하여 프로젝트 루트의 .env에서 환경변수를 로드합니다.
# load_dotenv는 이미 환경에 설정된 변수를 덮어쓰지 않습니다 (override=False).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 미설치 시 무시

# ── tiktoken (optional) ──────────────────────────────────────────────────
# 정확한 토큰 카운팅을 위해 로드합니다.
# 설치되지 않은 경우 한국어 비율 기반 가중치 폴백을 사용합니다.
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

# ── kiwipiepy (optional) — 한국어 형태소 분석기 ─────────────────────────
# 설치되면 extract_keywords()에서 정밀 형태소 분석을 수행합니다.
# 미설치 시 정규식 기반 조사 제거 폴백을 사용합니다.
try:
    from kiwipiepy import Kiwi
    _KIWI_AVAILABLE = True
    _kiwi_instance: Optional[Any] = None  # 지연 초기화 (lazy init)
except ImportError:
    _KIWI_AVAILABLE = False
    _kiwi_instance = None


def _get_kiwi() -> Any:
    """
    kiwipiepy Kiwi 인스턴스를 지연 초기화하여 반환합니다.
    [FIX-NEW-11] 싱글톤 패턴으로 Kiwi 객체를 재사용하여 초기화 오버헤드 제거
    
    [WARNING-P0] 이 코드를 multiprocessing.Process로 실행하면 Kiwi 인스턴스가
    pickle 불가능하여 실패합니다. 병렬화가 필요한 경우:
    1) 각 프로세스에서 독립적으로 Kiwi() 생성
    2) 또는 threading.Thread 사용 (multiprocessing 대신)
    """
    global _kiwi_instance
    if _kiwi_instance is None and _KIWI_AVAILABLE:
        _kiwi_instance = Kiwi()
    return _kiwi_instance


# External dependencies
try:
    from anthropic import Anthropic
    from openai import OpenAI
    # [FIX-CRITICAL-P0] Google SDK migration (Grok, Gemini 제안)
    # google-generativeai (deprecated 2025-11-30) → google-genai (신규 SDK)
    try:
        from google import genai
        from google.genai import types as genai_types
        _GOOGLE_NEW_SDK = True
    except ImportError:
        # Fallback to old SDK (will be removed in v6.0.0)
        import google.generativeai as genai
        _GOOGLE_NEW_SDK = False
        import logging
        logging.warning(
            "Using deprecated google-generativeai SDK. "
            "Please upgrade: pip install --upgrade google-genai"
        )
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run: pip install anthropic openai google-genai")
    exit(1)



# ============================================================================
# Global Thread Pool (스레드 메모리 누수 방지)
# ============================================================================
# [FIX-CRITICAL-P0] 매번 ThreadPoolExecutor를 생성/폐기하면 장기 실행 시
# 메모리 누수 발생. 전역 풀을 재사용하여 해결.
# Gemini, Grok 제안 반영

_GLOBAL_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="debate_worker"
)

def _cleanup_thread_pool():
    """
    프로그램 종료 시 스레드 풀 정리
    [FIX-CRITICAL-P0] atexit 대신 명시적 try-finally 사용 (Gemini 제안)
    레이스 컨디션 방지
    """
    global _GLOBAL_THREAD_POOL
    try:
        # 대기 중인 작업 완료까지 최대 5초 대기
        _GLOBAL_THREAD_POOL.shutdown(wait=True, timeout=5.0)
    except Exception as e:
        logging.getLogger(__name__).error(f"Thread pool cleanup error: {e}")
        # 강제 종료
        _GLOBAL_THREAD_POOL.shutdown(wait=False)

# ============================================================================
# Per-Turn Timeout Enforcement (스레드 기반 타임아웃 래퍼)
# ============================================================================
# config.turn_timeout 값이 있었지만 실제로 적용되지 않았습니다.
# API 호출이 무한히 걸리면 프로세스가 영원히 멈춥니다.
#
# [FIX-3] 타임아웃 후 executor.shutdown(wait=True)로 백그라운드 스레드가
#         완료될 때까지 메인 스레드가 블리킹되는 문제를 수정했습니다.
#         shutdown(wait=False)와 future.cancel()을 사용하여 즉시 반환하고,
#         백그라운드 스레드는 daemon 스레드로 실행되어 프로세스 종료 시
#         자동 정리됩니다.

class TurnTimeoutError(Exception):
    """단일 에이전트 턴이 turn_timeout을 초과한 경우 발생."""
    pass


def call_with_timeout(func, timeout_seconds: int, *args, **kwargs):
    """
    Execute func(*args, **kwargs) with a hard timeout.

    타임아웃 발생 시 백그라운드 스레드는 Python 제한으로 강제 종료되지 않지만,
    daemon=True 스레드로 실행되어 프로세스 종료 시 자동 정리됩니다.
    메인 루프는 즉시 TurnTimeoutError를 받아 다음 턴으로 진행합니다.
    """
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="timeout_worker"
    )
    future = executor.submit(func, *args, **kwargs)
    try:
        result = future.result(timeout=timeout_seconds)
        return result
    except concurrent.futures.TimeoutError:
        # future.cancel(): 아직 실행 시작 안된 경우에만 효과
        future.cancel()
        raise TurnTimeoutError(
            f"Turn exceeded {timeout_seconds}s timeout"
        )
    finally:
        # wait=False: 백그라운드 스레드 완료 여부와 무관하게 즉시 반환
        executor.shutdown(wait=False)


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
    """
    재시도 후에도 복구 불가 → 해당 토론자만 스킵.
    [FIX-NEW-8] reason 필드 추가로 스킵 사유 명시
    """
    def __init__(self, message: str, reason: str = "unknown"):
        super().__init__(message)
        self.reason = reason


def _is_transient_error(exc: Exception) -> bool:
    """예외 메시지가 일시적 오류 패턴과 일치하는지 판단."""
    msg = str(exc).lower()
    return any(kw in msg for kw in _TRANSIENT_KEYWORDS)


def call_with_retry(func, timeout_seconds: int, *args, **kwargs) -> Any:
    """
    call_with_timeout 위에 지수 백오프 재시도 레이어.
    TurnTimeoutError는 재시도하지 않고 즉시 상위로 전파됩니다.
    일시적 오류만 재시도하며, 영구 오류는 즉시 DebaterSkippedError로 변환됩니다.
    
    [FIX-NEW-8] API Key 오류, Quota 초과 등을 명확히 구분하여 reason 기록
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
            error_msg = str(exc).lower()
            
            # API Key 오류 즉시 실패
            if any(kw in error_msg for kw in ["api key", "authentication", "unauthorized", "401"]):
                raise DebaterSkippedError(
                    f"API authentication failed: {exc}",
                    reason="api_key_error"
                ) from exc
            
            # Quota 초과 즉시 실패
            if any(kw in error_msg for kw in ["quota", "billing", "exceeded"]):
                raise DebaterSkippedError(
                    f"API quota exceeded: {exc}",
                    reason="quota_exceeded"
                ) from exc
            
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
        f"Agent call failed after {MAX_RETRIES + 1} attempts",
        reason="max_retries_exceeded"
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
    debaters: List[AgentConfig] = field(default_factory=list)
    referee: Optional[AgentConfig] = None


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
    metadata: Dict[str, Any] = field(default_factory=dict)


class RefereeDecisionType(Enum):
    CORRECT         = "CORRECT"
    FACTUAL_ERROR   = "FACTUAL_ERROR"
    UNVERIFIABLE    = "UNVERIFIABLE"
    MISLEADING      = "MISLEADING"
    NEEDS_CONTEXT   = "NEEDS_CONTEXT"


@dataclass
class RefereeDecision:
    round_number: int
    turn_number: int
    target_debater: str
    claim: str
    decision: RefereeDecisionType
    reasoning: str
    evidence: List[str] = field(default_factory=list)
    correction: Optional[str] = None


@dataclass
class HallucinationAnnotation:
    turn_number: int
    debater_name: str
    flagged_claim: str
    hallucination_type: RefereeDecisionType
    correction_provided: bool
    correction_text: Optional[str] = None


@dataclass
class ExperimentResults:
    config: ExperimentConfig
    turns: List[Turn]
    hallucination_annotations: List[HallucinationAnnotation]
    referee_decisions: List[RefereeDecision]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Utility Functions
# ============================================================================

JACCARD_THRESHOLD = 0.6


def _estimate_tokens(text: str) -> int:
    """
    텍스트에서 토큰 수를 추정합니다.
    
    [FIX-4] 토큰 추정 승수를 1.3 → 1.5로 변경하여 보수적 추정
    """
    if not text or not text.strip():
        return 0
    return int(len(text.split()) * 1.5)


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
    텍스트에서 키워드를 추출합니다.

    우선순위:
      1) kiwipiepy 설치 시 → 형태소 분석 (명사만 추출, 최소 2자)
      2) kiwipiepy 미설치 시 → 정규식 조사 제거 폴백 (최소 2자)

    [FIX-1] 한국어 조사가 붙은 단어를 단순 공백 분리로만 처리하면
    "민주주의가" != "민주주의를" 으로 되어 Jaccard 유사도가 낮아지면서
    교착상태가 거의 탐지되지 않는 문제를 해결했습니다.
    """
    if not text or not text.strip():
        return frozenset()

    normalized = normalize_text(text)
    if not normalized:
        return frozenset()

    kiwi = _get_kiwi()

    if kiwi is not None:
        # ── 경로 A: kiwipiepy 형태소 분석 ──────────────────────────────
        # tag 접두사 'N' = 명사 (일반명사 NNG, 고유명사 NNP, 단위명사 NNB 등)
        try:
            tokens = kiwi.tokenize(normalized)
            return frozenset(
                t.form for t in tokens
                if t.tag.startswith('N') and len(t.form) >= 2
            )
        except Exception:
            pass  # kiwipiepy 실패 시 폴백으로 통과

    # ── 경로 B: 정규식 조사 제거 폴백 ──────────────────────────────────
    words = normalized.split()
    cleaned = []
    for w in words:
        # 조사 패턴이 단어 끝에 붙어 있으면 제거
        stripped = re.sub(r'(이|가|은|는|을|를|와|과|의|도|만|에서|로|으로|부터|까지|에게|한테|께|에|로써)$', '', w)
        if len(stripped) >= 2:
            cleaned.append(stripped)
    return frozenset(cleaned)


def jaccard_similarity(set1: frozenset, set2: frozenset) -> float:
    """
    두 집합 간의 Jaccard 유사도를 계산합니다.
    1.0 = 완전히 동일, 0.0 = 겹치는 요소 없음
    """
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    union        = set1 | set2
    return len(intersection) / len(union) if union else 0.0


def _serialize(obj: Any) -> Any:
    """
    Dataclass 객체를 JSON 직렬화 가능한 dict로 변환합니다.
    Enum은 .value로 변환하고, datetime은 ISO 문자열로, 재귀적으로 처리합니다.
    
    [FIX-NEW-9] dataclasses.asdict를 사용하여 안전하고 완전한 직렬화 보장
    [FIX-CRITICAL-NEW-2] datetime 객체 ISO 문자열 변환 추가
    """
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dataclass_fields__'):
        # dataclass 객체
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    else:
        return obj


# ============================================================================
# Agent Classes
# ============================================================================

class BaseAgent:
    """모든 LLM 에이전트의 기본 클래스"""

    def __init__(self, config: AgentConfig):
        self.config = config

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

    def reset(self):
        """Reset internal state if any"""
        pass


class ClaudeAgent(BaseAgent):
    """Anthropic Claude API를 사용하는 에이전트"""

    _HISTORY_LIMIT = 20  # 최대 히스토리 메시지 수 (페어 기준)

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key.strip() == "" or api_key.startswith("your_"):
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set, empty, or is a placeholder. "
                "Please set it in your .env file or as an environment variable."
            )

        self.client = Anthropic(api_key=api_key)
        self.conversation_history: List[Dict[str, str]] = []

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        [FIX-HIGH-P1] Debater API에도 재시도 로직 추가 (Gemini 제안)
        일시적 오류(429, 500, 503)로 전체 실험이 중단되는 것을 방지
        """
        start_time = time.time()
        last_exception = None
        
        for attempt in range(3):  # 최대 3회 재시도
            try:
                messages = self.conversation_history + [{"role": "user", "content": prompt}]

                response = self.client.messages.create(
                    model=self.config.model,
                    system=self.config.system_prompt,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    timeout=30.0
                )
                latency_ms = (time.time() - start_time) * 1000
                content = response.content[0].text

                self.conversation_history.append({"role": "user",      "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": content})
                if len(self.conversation_history) > self._HISTORY_LIMIT:
                    self.conversation_history = self.conversation_history[-self._HISTORY_LIMIT:]

                return {
                    "content": content,
                    "tokens": response.usage.input_tokens + response.usage.output_tokens,
                    "latency_ms": latency_ms,
                    "model": self.config.model
                }
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                # 일시적 오류인 경우만 재시도
                if attempt < 2 and any(keyword in error_msg for keyword in ['429', '500', '503', 'rate limit', 'overloaded']):
                    wait_time = (2 ** attempt)  # 지수 백오프: 1s, 2s
                    logging.getLogger(__name__).warning(
                        f"Claude API transient error (attempt {attempt+1}/3): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    break
        
        raise RuntimeError(f"Claude API error after retries: {str(last_exception)}") from last_exception


class GPTAgent(BaseAgent):
    """OpenAI GPT API를 사용하는 에이전트"""

    _HISTORY_LIMIT = 20

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.strip() == "" or api_key.startswith("your_"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set, empty, or is a placeholder. "
                "Please set it in your .env file or as an environment variable."
            )

        self.client = OpenAI(api_key=api_key)
        self.conversation_history: List[Dict[str, str]] = []

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        [FIX-HIGH-P1] Debater API에도 재시도 로직 추가 (Gemini 제안)
        """
        start_time = time.time()
        last_exception = None
        
        for attempt in range(3):
            try:
                messages = (
                    [{"role": "system", "content": self.config.system_prompt}]
                    + self.conversation_history
                    + [{"role": "user", "content": prompt}]
                )

                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    timeout=30.0
                )
                latency_ms = (time.time() - start_time) * 1000
                content = response.choices[0].message.content

                self.conversation_history.append({"role": "user",      "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": content})
                if len(self.conversation_history) > self._HISTORY_LIMIT:
                    self.conversation_history = self.conversation_history[-self._HISTORY_LIMIT:]

                return {
                    "content": content,
                    "tokens": response.usage.total_tokens,
                    "latency_ms": latency_ms,
                    "model": self.config.model
                }
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                if attempt < 2 and any(keyword in error_msg for keyword in ['429', '500', '503', 'rate limit', 'overloaded']):
                    wait_time = (2 ** attempt)
                    logging.getLogger(__name__).warning(
                        f"GPT API transient error (attempt {attempt+1}/3): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    break
        
        raise RuntimeError(f"GPT API error after retries: {str(last_exception)}") from last_exception


class GeminiReferee(BaseAgent):
    """
    Referee agent using Google's Gemini — stateless design.

    [v5.3.0] 신규 google-genai SDK 기반으로 재작성 (Grok, Gemini 제안)
    - 구 SDK(google-generativeai) fallback 지원
    - system_instruction 예외 처리 강화
    - 2026년 표준 API 사용
    """
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key.strip() == "" or api_key.startswith("your_"):
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set, empty, or is a placeholder. "
                "Please set it in your .env file or as an environment variable."
            )

        logger = logging.getLogger(__name__)
        self.system_prompt_in_model = False
        
        try:
            if _GOOGLE_NEW_SDK:
                # === 신규 SDK (google-genai) ===
                self.client = genai.Client(api_key=api_key)
                self.model_id = config.model
                
                # system_instruction 지원 확인
                try:
                    self.gen_config = genai_types.GenerateContentConfig(
                        system_instruction=config.system_prompt,
                        temperature=config.temperature,
                        max_output_tokens=config.max_tokens,
                        response_mime_type="application/json"
                    )
                    self.system_prompt_in_model = True
                    logger.info(f"Gemini model '{config.model}' initialized with system_instruction (new SDK)")
                except Exception as e:
                    # system_instruction 미지원 모델
                    logger.warning(
                        f"Model '{config.model}' does not support system_instruction: {e}. "
                        f"Falling back to prepending system prompt."
                    )
                    self.gen_config = genai_types.GenerateContentConfig(
                        temperature=config.temperature,
                        max_output_tokens=config.max_tokens,
                        response_mime_type="application/json"
                    )
                    self.system_prompt_in_model = False
            else:
                # === 구 SDK (google-generativeai) fallback ===
                genai.configure(api_key=api_key)
                try:
                    self.model = genai.GenerativeModel(
                        model_name=config.model,
                        system_instruction=config.system_prompt
                    )
                    self.system_prompt_in_model = True
                    logger.warning(
                        f"Using deprecated google-generativeai SDK. "
                        f"Please upgrade to google-genai for full support."
                    )
                except (ValueError, TypeError, AttributeError, Exception) as e:
                    # [FIX-HIGH-P1] 모든 예외 타입 포괄 (Gemini 제안)
                    error_msg = str(e).lower()
                    if "system_instruction" in error_msg or "unsupported" in error_msg or "not supported" in error_msg:
                        logger.warning(
                            f"Model '{config.model}' does not support system_instruction: {e}. "
                            f"Falling back."
                        )
                        self.model = genai.GenerativeModel(model_name=config.model)
                        self.system_prompt_in_model = False
                    else:
                        raise
        except Exception as e:
            logger.critical(f"Failed to initialize Gemini model '{config.model}': {e}")
            raise RuntimeError(f"Gemini model initialization failed: {e}") from e

        self.decision_log: deque = deque(maxlen=20)

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        [v5.3.0] 신규/구 SDK 모두 지원하는 generate_response
        """
        start_time = time.time()
        decision_log_text = self._format_decision_log()
        
        # system_instruction 미지원 시 프롬프트에 포함
        if not self.system_prompt_in_model:
            system_prefix = f"[SYSTEM INSTRUCTION]\n{self.config.system_prompt}\n\n"
            full_prompt = f"{system_prefix}{decision_log_text}\n\n{prompt}" if decision_log_text else f"{system_prefix}{prompt}"
        else:
            full_prompt = f"{decision_log_text}\n\n{prompt}" if decision_log_text else prompt

        try:
            if _GOOGLE_NEW_SDK:
                # === 신규 SDK 호출 ===
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=full_prompt,
                    config=self.gen_config
                )
                latency_ms = (time.time() - start_time) * 1000
                
                # 응답 텍스트 추출
                if hasattr(response, 'text'):
                    content_text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    content_text = response.candidates[0].content.parts[0].text
                else:
                    raise ValueError("Cannot extract text from Gemini response")
                
                # 토큰 카운트 (신규 SDK)
                try:
                    count_result = self.client.models.count_tokens(
                        model=self.model_id,
                        contents=content_text
                    )
                    tokens = count_result.total_tokens
                except Exception:
                    tokens = _estimate_tokens(content_text)
                
                return {
                    "content": content_text,
                    "tokens": tokens,
                    "latency_ms": latency_ms,
                    "model": self.config.model,
                    "grounding_metadata": getattr(response, 'grounding_metadata', None)
                }
            else:
                # === 구 SDK 호출 (fallback) ===
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_tokens,
                        response_mime_type="application/json"
                    )
                )
                latency_ms = (time.time() - start_time) * 1000
                content_text = response.text

                try:
                    token_result = self.model.count_tokens(content_text)
                    tokens = token_result.total_tokens
                except Exception:
                    tokens = _estimate_tokens(content_text)

                return {
                    "content": content_text,
                    "tokens": tokens,
                    "latency_ms": latency_ms,
                    "model": self.config.model,
                    "grounding_metadata": getattr(response, 'grounding_metadata', None)
                }
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}") from e

    def _format_decision_log(self) -> str:
        """
        [FIX-NEW-10] 판정 로그를 요약 형태로 제공하여 컨텍스트 윈도우 낭비 방지
        과거 판정을 통째로 넘기지 않고, 핵심 패턴만 추출
        """
        if not self.decision_log:
            return ""
        
        # 최근 5개의 결정에서 패턴 추출
        recent_decisions = list(self.decision_log)[-5:]
        error_types = {}
        for dec in recent_decisions:
            dec_type = dec.decision.value if isinstance(dec.decision, Enum) else str(dec.decision)
            error_types[dec_type] = error_types.get(dec_type, 0) + 1
        
        log_text = "[DECISION GUIDANCE]\n"
        log_text += "Recent error patterns (for consistency):\n"
        for error_type, count in error_types.items():
            log_text += f"- {error_type}: {count} cases\n"
        
        # 가장 최근 2건만 상세 표시
        log_text += "\nMost recent decisions:\n"
        for dec in recent_decisions[-2:]:
            log_text += f"Round {dec.round_number}: {dec.target_debater} → {dec.decision.value if isinstance(dec.decision, Enum) else dec.decision}\n"
        
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

    def _create_fallback_turn(self, round_num: int, turn_num: int,
                              agent: BaseAgent, role: str, status_content: str,
                              skip_reason: str = "unknown") -> Turn:
        """
        [FIX-2] Timeout / Skipped / Error 시 Turn 객체를 생성하여
        full_transcript.json에 빈 갭 없이 기록합니다.
        [FIX-NEW-8] skip_reason 추가로 실패 원인 명시
        """
        return Turn(
            round_number=round_num,
            turn_number=turn_num,
            agent_role=role,
            agent_name=agent.config.name,
            model=agent.config.model,
            timestamp=datetime.now().isoformat(),
            content=status_content,
            tokens_used=0,
            latency_ms=0.0,
            metadata={"status": status_content, "fallback": True, "skip_reason": skip_reason}
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
            # fallback 턴은 짧은 상태 메시지이므로 그대로 표시
            content_summary = (turn.content[:200] + "..."
                               if len(turn.content) > 200
                               else turn.content)
            context += f"{tag}\n{content_summary}\n\n"
        return context

    def run_debate(self, initial_prompt: str) -> Tuple[List[Turn], List[RefereeDecision]]:
        """
        라운드 기반 토론 실행.
        각 에이전트 호출에 turn_timeout을 스레드 타임아웃으로 적용합니다.

        [FIX-2] TurnTimeoutError / DebaterSkippedError / 기타 Exception 발생
        시에도 Turn 객체를 생성하여 self.turns에 기록하여, full_transcript.json
        에서 turn_number 갭이 생기지 않도록 했습니다.
        """
        self.logger.info(f"Starting debate: {self.config.topic}")
        turn_counter = 0

        for round_num in range(1, self.config.max_rounds + 1):
            self.logger.info("=" * 80)
            self.logger.info(f"ROUND {round_num} START")
            self.logger.info("=" * 80)

            round_statements: List[Tuple[str, str, int]] = []

            # ── 각 토론자 차례 ──────────────────────────────────────────
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
                    response = call_with_retry(
                        debater.generate_response,
                        self.config.turn_timeout,
                        prompt
                    )
                    turn = self._create_turn(
                        round_num, turn_counter, debater, response, debater_role
                    )
                    self.turns.append(turn)
                    round_statements.append((
                        debater.config.name, response["content"], turn_counter
                    ))
                    self._detect_deadlock(response["content"], debater.config.name)

                except TurnTimeoutError:
                    self.logger.error(
                        f"{debater.config.name} timed out (>{self.config.turn_timeout}s)"
                    )
                    turn = self._create_fallback_turn(
                        round_num, turn_counter, debater, debater_role,
                        f"[TIMEOUT: exceeded {self.config.turn_timeout}s]",
                        skip_reason="timeout"
                    )
                    self.turns.append(turn)

                except DebaterSkippedError as e:
                    self.logger.error(f"{debater.config.name} skipped: {e}")
                    turn = self._create_fallback_turn(
                        round_num, turn_counter, debater, debater_role,
                        f"[SKIPPED: {str(e)}]",
                        skip_reason=getattr(e, 'reason', 'unknown')
                    )
                    self.turns.append(turn)

                except Exception as e:
                    self.logger.error(
                        f"Unexpected error for {debater.config.name}: {type(e).__name__}: {e}"
                    )
                    turn = self._create_fallback_turn(
                        round_num, turn_counter, debater, debater_role,
                        f"[ERROR: {type(e).__name__}]",
                        skip_reason="unexpected_error"
                    )
                    self.turns.append(turn)

            # ── 그룹 교착 확인 (라운드 종료 후) ──────────────────────────
            if self._check_group_deadlock():
                self.logger.warning(
                    f"Group deadlock detected at round {round_num}. "
                    f"Stopping debate early."
                )
                break

            # ── Referee 호출 (라운드별) ─────────────────────────────────
            if not round_statements:
                self.logger.warning(f"Round {round_num} has no statements → skip referee")
                continue

            turn_counter += 1
            referee_prompt = self._build_referee_prompt(round_num, round_statements)

            try:
                referee_response = call_with_timeout(
                    self.referee.generate_response,
                    self.config.turn_timeout,
                    referee_prompt
                )
                turn = self._create_turn(
                    round_num, turn_counter, self.referee, referee_response, "referee"
                )
                self.turns.append(turn)

                decisions = self._extract_and_log_decisions(
                    round_num, referee_response["content"]
                )
                if decisions:
                    self.all_referee_decisions.extend(decisions)
                    if isinstance(self.referee, GeminiReferee):
                        for dec in decisions:
                            self.referee.add_decision(dec)

            except TurnTimeoutError:
                self.logger.error(
                    f"Referee timed out (>{self.config.turn_timeout}s)"
                )
                dummy_content = f"[TIMEOUT: exceeded {self.config.turn_timeout}s]"
                turn = self._create_fallback_turn(
                    round_num, turn_counter, self.referee, "referee",
                    dummy_content, skip_reason="timeout"
                )
                self.turns.append(turn)
                # [FIX-NEW-7] Referee fallback 시에도 빈 decision 기록으로 metrics 왜곡 방지
                # timeout으로 판정하지 못했음을 명시
                self.logger.warning(f"Round {round_num} referee decisions unavailable (timeout)")

            except Exception as e:
                self.logger.error(f"Referee error: {type(e).__name__}: {e}")
                dummy_content = f"[ERROR: {type(e).__name__}]"
                turn = self._create_fallback_turn(
                    round_num, turn_counter, self.referee, "referee",
                    dummy_content, skip_reason="unexpected_error"
                )
                self.turns.append(turn)
                self.logger.warning(f"Round {round_num} referee decisions unavailable (error)")

        self.logger.info(f"Debate ended. Total turns: {len(self.turns)}")
        return self.turns, self.all_referee_decisions

    def _build_debater_prompt(self, round_num: int, debater_idx: int,
                              debater_name: str, round_statements: List[Tuple[str, str, int]],
                              initial_prompt: str) -> str:
        context = self._get_recent_context()
        current_round_context = ""

        if round_statements:
            current_round_context = f"\n[CURRENT ROUND {round_num} STATEMENTS]\n"
            for name, stmt, turn_num in round_statements:
                current_round_context += f"[{name}] (Turn {turn_num}):\n{stmt}\n\n"

        return (
            f"{context}"
            f"{current_round_context}"
            f"\n{debater_name}, present your argument.\n"
            f"Task: {initial_prompt}\n\n"
            f"IMPORTANT: Reference specific statements from other debaters by their turn numbers."
        )

    def _build_referee_prompt(self, round_num: int,
                              round_statements: List[Tuple[str, str, int]]) -> str:
        statements_text = ""
        for name, stmt, turn_num in round_statements:
            statements_text += (
                f"──────────────────────────────────────\n"
                f"Debater: {name}\n"
                f"Turn: {turn_num}\n"
                f"Statement:\n{stmt}\n\n"
            )

        # [FIX-NEW-CRITICAL-12] JSON 출력 명시 추가로 Gemini JSON mode 충돌 방지
        return f"""You are a critical fact-checking referee. Analyze ALL claims in Round {round_num}.

{statements_text}

For EACH claim that requires fact-checking, you MUST respond in valid JSON array format.
Output must be a valid JSON object.

[
  {{
    "round_number": {round_num},
    "turn_number": <exact turn number>,
    "target_debater": "<exact debater name>",
    "claim": "<exact quote from statement>",
    "decision": "<CORRECT|FACTUAL_ERROR|UNVERIFIABLE|MISLEADING|NEEDS_CONTEXT>",
    "reasoning": "<your detailed reasoning>",
    "evidence": ["<source1>", "<source2>"],
    "correction": "<correction if decision is not CORRECT, otherwise null>"
  }}
]

CRITICAL RULES:
1. Response MUST be valid JSON array (not object)
2. If all claims are correct, return empty array: []
3. decision must be EXACTLY one of: CORRECT, FACTUAL_ERROR, UNVERIFIABLE, MISLEADING, NEEDS_CONTEXT
4. evidence should be real sources when possible
5. correction is required for FACTUAL_ERROR, MISLEADING, NEEDS_CONTEXT
6. Output ONLY the JSON array, no additional text.
7. Ensure all JSON is properly formatted with correct quotes and escaping."""

    def _extract_and_log_decisions(self, round_num: int, 
                                   referee_content: str) -> List[RefereeDecision]:
        """
        심판의 JSON 응답을 파싱하여 RefereeDecision 리스트로 변환합니다.
        
        [FIX-NEW-10] JSON mode 강제로 파싱 실패율 감소, 실패 시 상세 로깅
        """
        try:
            # JSON 전처리: markdown 코드 블록 제거
            cleaned = referee_content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            decisions_raw = json.loads(cleaned)
            
            if not isinstance(decisions_raw, list):
                self.logger.error(
                    f"Referee response is not a JSON array. Got: {type(decisions_raw)}"
                )
                return []

            decisions = []
            for item in decisions_raw:
                try:
                    # decision 문자열을 Enum으로 변환
                    decision_str = item.get("decision", "").upper()
                    try:
                        decision_type = RefereeDecisionType[decision_str]
                    except KeyError:
                        self.logger.warning(
                            f"Unknown decision type '{decision_str}', defaulting to NEEDS_CONTEXT"
                        )
                        decision_type = RefereeDecisionType.NEEDS_CONTEXT

                    dec = RefereeDecision(
                        round_number=item.get("round_number", round_num),
                        turn_number=item.get("turn_number", 0),
                        target_debater=item.get("target_debater", "Unknown"),
                        claim=item.get("claim", ""),
                        decision=decision_type,
                        reasoning=item.get("reasoning", ""),
                        evidence=item.get("evidence", []),
                        correction=item.get("correction")
                    )
                    decisions.append(dec)
                except Exception as e:
                    self.logger.error(f"Failed to parse decision item: {e}\nItem: {item}")
                    continue

            self.logger.info(
                f"Round {round_num}: Extracted {len(decisions)} referee decisions"
            )
            return decisions

        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to parse referee JSON in round {round_num}: {e}\n"
                f"Content: {referee_content[:500]}"
            )
            return []
        except Exception as e:
            self.logger.error(
                f"Unexpected error extracting referee decisions: {e}"
            )
            return []


# ============================================================================
# Hallucination Evaluator
# ============================================================================

class HallucinationEvaluator:
    """심판 판정을 기반으로 환각 통계를 생성합니다."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def extract_referee_annotations(self, turns: List[Turn],
                                    decisions: List[RefereeDecision]
                                    ) -> List[HallucinationAnnotation]:
        annotations = []
        for dec in decisions:
            if dec.decision == RefereeDecisionType.CORRECT:
                continue

            annotations.append(HallucinationAnnotation(
                turn_number=dec.turn_number,
                debater_name=dec.target_debater,
                flagged_claim=dec.claim,
                hallucination_type=dec.decision,
                correction_provided=(dec.correction is not None and dec.correction.strip() != ""),
                correction_text=dec.correction
            ))
        return annotations

    def calculate_metrics(self, turns: List[Turn],
                         annotations: List[HallucinationAnnotation]) -> Dict[str, Any]:
        # fallback 턴 제외
        debater_turns = [t for t in turns if t.agent_role.startswith("debater")
                         and not t.metadata.get("fallback", False)]
        referee_turns = [t for t in turns if t.agent_role == "referee"
                        and not t.metadata.get("fallback", False)]

        total_debater_turns = len(debater_turns)
        hallucination_count  = len(annotations)

        hallucination_rate = (hallucination_count / total_debater_turns
                              if total_debater_turns > 0 else 0.0)

        corrections_provided = sum(1 for a in annotations if a.correction_provided)
        correction_rate = (corrections_provided / hallucination_count
                          if hallucination_count > 0 else 0.0)

        type_counts = {
            "factual_errors":     0,
            "unverifiable_claims": 0,
            "misleading_claims":   0,
            "needs_context":       0
        }
        for ann in annotations:
            if ann.hallucination_type == RefereeDecisionType.FACTUAL_ERROR:
                type_counts["factual_errors"] += 1
            elif ann.hallucination_type == RefereeDecisionType.UNVERIFIABLE:
                type_counts["unverifiable_claims"] += 1
            elif ann.hallucination_type == RefereeDecisionType.MISLEADING:
                type_counts["misleading_claims"] += 1
            elif ann.hallucination_type == RefereeDecisionType.NEEDS_CONTEXT:
                type_counts["needs_context"] += 1

        return {
            "total_turns":           len(turns),
            "debater_turns":         total_debater_turns,
            "referee_interventions": len(referee_turns),
            "hallucination_rate":    hallucination_rate,
            "correction_rate":       correction_rate,
            "hallucinations_detected": hallucination_count,
            "corrections_provided":  corrections_provided,
            **type_counts
        }


# ============================================================================
# Experiment Runner
# ============================================================================

class ExperimentRunner:
    """실험 전체 파이프라인을 관리합니다."""

    def __init__(self, experiment_name: str, num_debaters: int,
                 seed: int, output_base: str):
        self.experiment_name = experiment_name
        self.num_debaters    = num_debaters
        self.seed            = seed
        self.timestamp       = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        random.seed(seed)

        exp_id = f"{experiment_name}_{num_debaters}d_{self.timestamp}"
        self.output_dir = os.path.join(output_base, exp_id)
        
        # [FIX-NEW-CRITICAL-13] output 디렉토리를 logger 설정 전에 생성
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"referee_experiment_{self.experiment_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        log_file = os.path.join(self.output_dir, "debate.log")
        # [FIX-5] 로그 파일 크기 증가 (5MB → 20MB, backup 5개)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=20*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

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

        # [FIX-6] 같은 stance의 토론자가 여러 명일 때 이름에 인덱스 접미사를 추가하여
        #   per_debater_keyword_history의 키 충돌을 방지합니다.
        #   (충돌 시 같은 키워드 히스토리를 공유하여 교착 탐지 오작동)
        stance_counter: Dict[Stance, int] = {}  # 각 stance별 등장 횟수

        debaters = []
        for i in range(self.num_debaters):
            stance = stances[i] if i < len(stances) else Stance.NEUTRAL
            model  = models[i % len(models)]
            base_name, prompt = stance_templates[stance]

            # 같은 stance가 2회 이상 등장할 때만 접미사 추가
            stance_counter[stance] = stance_counter.get(stance, 0) + 1
            count = stance_counter[stance]
            # 해당 stance의 총 등장 횟수 미리 계산
            total_of_stance = stances.count(stance)
            if total_of_stance > 1:
                name = f"{base_name} #{count}"
            else:
                name = base_name

            debaters.append(AgentConfig(
                name=name, role=AgentRole.DEBATER, model=model,
                temperature=0.7, max_tokens=1000,
                system_prompt=prompt, stance=stance
            ))
        return debaters

    def _get_experiment_config(self, experiment_name: str) -> ExperimentConfig:
        debaters = self._create_balanced_debaters(experiment_name)
        
        # [FIX-NEW-CRITICAL-12] Gemini 프롬프트에 JSON 출력 명시 추가
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

Respond ONLY in valid JSON format with the structure specified.
Output must be a valid JSON object."""
        )

        # [FIX-7] max_context_turns를 토론자 수에 따라 동적 조정합니다.
        #   debater 수가 8명이면 1라운드 = 8턴 + 1심판 = 9턴.
        #   고정 10으로는 이전 라운드 컨텍스트가 거의 남지 않습니다.
        #   공식: max(10, num_debaters * 3)
        dynamic_max_context = max(10, len(debaters) * 3)

        return ExperimentConfig(
            experiment_id=f"{experiment_name}_{self.num_debaters}d_{self.timestamp}",
            topic=experiment_name,
            description=f"ML training data: {self.num_debaters} debaters",
            max_rounds=5,
            turn_timeout=60,
            deadlock_threshold=3,
            seed=self.seed,
            timestamp=self.timestamp,
            max_context_turns=dynamic_max_context,
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
        """
        [FIX-12] _serialize() 헬퍼를 사용하여 Enum을 .value 문자열로 직렬화합니다.
        """
        files = {
            "full_transcript.json":           [_serialize(t) for t in results.turns],
            "referee_decisions.json":         [_serialize(d) for d in results.referee_decisions],
            "hallucination_annotations.json": [_serialize(a) for a in results.hallucination_annotations],
            "metrics.json":                   results.metrics,
        }
        for filename, data in files.items():
            path = os.path.join(self.output_dir, filename)
            with open(path, 'w', encoding='utf-8') as f:
                # [FIX-MEDIUM-P2] default=str 제거 (_serialize가 처리)
                json.dump(data, f, indent=2, ensure_ascii=False)

        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            # [FIX-MEDIUM-P2] default=str 제거
            json.dump(_serialize(results.config), f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to {self.output_dir}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    [FIX-MEDIUM-P2] outputs 디렉토리 사전 생성 (Grok 제안)
    로컬 실행 시에도 디렉토리 생성 보장
    """
    # 기본 outputs 디렉토리 생성
    os.makedirs("outputs", exist_ok=True)
    
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

    # [FIX-NEW-CRITICAL-7] 더 명확한 오류 메시지 - 조건별 분리
    if args.debaters < 4:
        print("ERROR: --debaters must be >= 4 (minimum 4 debaters required)")
        return 1
    if args.debaters % 2 != 0:
        print("ERROR: --debaters must be even (requires balanced debate sides)")
        return 1

    required_keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]
    
    # [FIX-CRITICAL-4] placeholder 검사 추가
    missing_or_placeholder = []
    for key in required_keys:
        value = os.getenv(key)
        if not value or value.startswith("your_"):
            missing_or_placeholder.append(key)

    if missing_or_placeholder:
        print("\n❌ Missing or placeholder API keys:")
        for key in missing_or_placeholder:
            print(f"   - {key}")
        print("\nPlease set valid API keys in your .env file or as environment variables.")
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
    try:
        exit_code = main()
    finally:
        # [FIX-CRITICAL-P0] 스레드 풀 명시적 정리 (Gemini 제안)
        _cleanup_thread_pool()
    exit(exit_code)

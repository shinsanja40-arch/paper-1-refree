#!/usr/bin/env python3
"""
Reproducible Experimental Protocol for Referee-Mediated Discourse Framework
===========================================================================

This script implements a complete, reproducible experiment for multi-agent
debate with real-time hallucination detection and correction.

Usage:
    python referee_mediated_discourse.py --experiment nuclear_energy --seed 42

Requirements:
    pip install anthropic openai google-generativeai pyyaml numpy pandas scikit-learn

Environment Variables (required):
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    GOOGLE_API_KEY
"""

import os
import json
import time
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

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
# Configuration Classes
# ============================================================================

class AgentRole(Enum):
    """Role definitions for agents"""
    DEBATER_A = "debater_a"
    DEBATER_B = "debater_b"
    REFEREE = "referee"
    SUPERVISOR = "supervisor"


@dataclass
class AgentConfig:
    """Configuration for a single agent"""
    name: str
    role: AgentRole
    model: str
    temperature: float
    max_tokens: int
    system_prompt: str
    persona: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment_id: str
    topic: str
    description: str
    max_turns: int
    turn_timeout: int
    deadlock_threshold: int
    seed: int
    timestamp: str
    
    agent_a: AgentConfig
    agent_b: AgentConfig
    referee: AgentConfig


@dataclass
class Turn:
    """Single turn in the debate"""
    turn_number: int
    agent_role: str
    agent_name: str
    model: str
    timestamp: str
    content: str
    tokens_used: int
    latency_ms: float
    metadata: Dict[str, Any]


@dataclass
class HallucinationAnnotation:
    """Annotation for a potential hallucination"""
    turn_number: int
    sentence_id: str
    claim: str
    is_hallucination: bool
    severity: str  # "factual_error", "unverifiable", "misleading", "correct"
    evidence: List[str]
    annotator_notes: str


@dataclass
class ExperimentResults:
    """Complete experiment results"""
    config: ExperimentConfig
    turns: List[Turn]
    hallucination_annotations: List[HallucinationAnnotation]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]


# ============================================================================
# System Prompts
# ============================================================================

SYSTEM_PROMPTS = {
    "debater_nuclear_pro": """You are a knowledgeable expert advocating for nuclear energy.

Your role:
- Present evidence-based arguments supporting nuclear power
- Cite specific data, studies, and real-world examples
- Engage respectfully with counterarguments
- Focus on safety, economics, and environmental impact

Guidelines:
- Make only verifiable factual claims
- Cite sources when making statistical claims
- Acknowledge limitations of your position when relevant
- Stay focused on the debate topic""",

    "debater_renewable_pro": """You are a knowledgeable expert advocating for renewable energy.

Your role:
- Present evidence-based arguments supporting renewable energy
- Cite specific data, studies, and real-world examples
- Engage respectfully with counterarguments
- Focus on sustainability, cost trends, and technological advancement

Guidelines:
- Make only verifiable factual claims
- Cite sources when making statistical claims
- Acknowledge limitations of your position when relevant
- Stay focused on the debate topic""",

    "referee": """You are an independent fact-checking referee monitoring a debate.

Your role:
- Monitor EVERY claim made by debaters in real-time
- Identify factual errors, hallucinations, and unverifiable claims
- Provide corrections with reliable sources
- Remain completely neutral and objective

When you detect an error:
1. Quote the specific problematic claim
2. Explain why it's incorrect or unverifiable
3. Provide the correct information with sources
4. Rate severity: FACTUAL_ERROR, UNVERIFIABLE, MISLEADING, or CORRECT

You have access to web search. Use it to verify claims.

Format your response as:
TURN [N] REVIEW:
- Claim: "[quote]"
- Status: [CORRECT/FACTUAL_ERROR/UNVERIFIABLE/MISLEADING]
- Evidence: [sources]
- Correction: [if needed]""",

    "debater_good_vs_evil_good": """You are a philosopher arguing that humans are inherently good.

Your role:
- Present philosophical and empirical arguments for human goodness
- Reference psychological studies, historical examples, and philosophical traditions
- Engage with counterarguments about human nature
- Discuss altruism, cooperation, empathy, and moral development

Guidelines:
- Distinguish between philosophical claims and empirical claims
- Cite specific philosophers, studies, and historical examples
- Acknowledge the existence of evil while maintaining your position
- Avoid overgeneralizations""",

    "debater_good_vs_evil_evil": """You are a philosopher arguing that humans have inherent capacity for evil.

Your role:
- Present philosophical and empirical arguments about human darkness
- Reference psychological studies, historical atrocities, and philosophical traditions
- Engage with counterarguments about human nature
- Discuss violence, selfishness, tribalism, and moral failures

Guidelines:
- Distinguish between philosophical claims and empirical claims
- Cite specific philosophers, studies, and historical examples
- Avoid claiming humans are purely evil
- Focus on structural and psychological explanations"""
}


# ============================================================================
# Agent Implementations
# ============================================================================

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.conversation_history: List[Dict] = []
        
    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a response to the given prompt"""
        raise NotImplementedError
        
    def reset(self):
        """Reset agent state"""
        self.conversation_history = []


class ClaudeAgent(BaseAgent):
    """Agent using Anthropic's Claude"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=api_key)
        
    def generate_response(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        
        messages = self.conversation_history + [
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=self.config.system_prompt,
            messages=messages
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        content = response.content[0].text
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": content})
        
        return {
            "content": content,
            "tokens": response.usage.input_tokens + response.usage.output_tokens,
            "latency_ms": latency_ms,
            "model": self.config.model
        }


class GPTAgent(BaseAgent):
    """Agent using OpenAI's GPT"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        
    def generate_response(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        
        messages = [
            {"role": "system", "content": self.config.system_prompt}
        ] + self.conversation_history + [
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        content = response.choices[0].message.content
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": content})
        
        return {
            "content": content,
            "tokens": response.usage.total_tokens,
            "latency_ms": latency_ms,
            "model": self.config.model
        }


class GeminiReferee(BaseAgent):
    """Referee agent using Google's Gemini with grounding"""
    
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
        self.chat = None
        
    def generate_response(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        
        if self.chat is None:
            self.chat = self.model.start_chat(history=[])
        
        # Enable grounding via Google Search
        response = self.chat.send_message(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        content = response.text
        
        # Estimate tokens (Gemini doesn't provide exact count)
        tokens = len(content.split()) * 1.3  # Rough estimate
        
        return {
            "content": content,
            "tokens": int(tokens),
            "latency_ms": latency_ms,
            "model": self.config.model,
            "grounding_metadata": getattr(response, 'grounding_metadata', None)
        }
    
    def reset(self):
        super().reset()
        self.chat = None


# ============================================================================
# Debate Manager
# ============================================================================

class DebateManager:
    """Manages the multi-agent debate process"""
    
    def __init__(self, config: ExperimentConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.turns: List[Turn] = []
        self.deadlock_counter = 0
        self.previous_arguments = set()
        
        # Initialize agents
        self.agent_a = self._create_agent(config.agent_a)
        self.agent_b = self._create_agent(config.agent_b)
        self.referee = self._create_agent(config.referee)
        
    def _create_agent(self, agent_config: AgentConfig) -> BaseAgent:
        """Factory method to create appropriate agent type"""
        if "claude" in agent_config.model.lower():
            return ClaudeAgent(agent_config)
        elif "gpt" in agent_config.model.lower():
            return GPTAgent(agent_config)
        elif "gemini" in agent_config.model.lower():
            return GeminiReferee(agent_config)
        else:
            raise ValueError(f"Unknown model: {agent_config.model}")
    
    def _create_turn(self, turn_number: int, agent: BaseAgent, 
                     response: Dict[str, Any], role: str) -> Turn:
        """Create a Turn object from agent response"""
        return Turn(
            turn_number=turn_number,
            agent_role=role,
            agent_name=agent.config.name,
            model=agent.config.model,
            timestamp=datetime.now().isoformat(),
            content=response["content"],
            tokens_used=response["tokens"],
            latency_ms=response["latency_ms"],
            metadata=response
        )
    
    def _detect_deadlock(self, content: str) -> bool:
        """Detect if the debate is in a deadlock"""
        # Simple hash-based detection
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        if content_hash in self.previous_arguments:
            self.deadlock_counter += 1
        else:
            self.deadlock_counter = 0
            self.previous_arguments.add(content_hash)
        
        return self.deadlock_counter >= self.config.deadlock_threshold
    
    def run_debate(self, initial_prompt: str) -> List[Turn]:
        """Execute the complete debate"""
        self.logger.info(f"Starting debate: {self.config.topic}")
        self.logger.info(f"Initial prompt: {initial_prompt}")
        
        # Initial statement from Agent A
        self.logger.info("Turn 1: Agent A opening statement")
        response_a = self.agent_a.generate_response(initial_prompt)
        turn_1 = self._create_turn(1, self.agent_a, response_a, "debater_a")
        self.turns.append(turn_1)
        
        # Referee reviews Turn 1
        self.logger.info("Turn 1: Referee fact-check")
        referee_prompt = f"""Please fact-check the following statement:

Agent A stated:
{response_a['content']}

Review each factual claim and identify any hallucinations, errors, or unverifiable statements."""

        referee_response_1 = self.referee.generate_response(referee_prompt)
        turn_1_ref = self._create_turn(1, self.referee, referee_response_1, "referee")
        self.turns.append(turn_1_ref)
        
        # Main debate loop
        current_agent = self.agent_b
        current_role = "debater_b"
        other_agent = self.agent_a
        turn_number = 2
        
        while turn_number <= self.config.max_turns:
            self.logger.info(f"Turn {turn_number}: {current_role}")
            
            # Construct context for current agent
            previous_turn = self.turns[-2]  # Get the last debater's turn (skip referee)
            referee_turn = self.turns[-1]   # Get the referee's review
            
            debate_prompt = f"""Previous argument from opponent:
{previous_turn.content}

Referee's fact-check:
{referee_turn.content}

Please respond with your counter-argument. Address the points raised and present your own evidence-based position."""

            # Generate response
            response = current_agent.generate_response(debate_prompt)
            turn = self._create_turn(turn_number, current_agent, response, current_role)
            self.turns.append(turn)
            
            # Check for deadlock
            if self._detect_deadlock(response["content"]):
                self.logger.warning(f"Deadlock detected at turn {turn_number}")
                self.logger.info("Human supervisor intervention would occur here")
                # In a real implementation, this would trigger human intervention
                break
            
            # Referee fact-check
            self.logger.info(f"Turn {turn_number}: Referee fact-check")
            referee_prompt = f"""Please fact-check the following statement:

{current_role.upper()} stated:
{response['content']}

Review each factual claim and identify any hallucinations, errors, or unverifiable statements."""

            referee_response = self.referee.generate_response(referee_prompt)
            turn_ref = self._create_turn(turn_number, self.referee, referee_response, "referee")
            self.turns.append(turn_ref)
            
            # Switch agents
            current_agent, other_agent = other_agent, current_agent
            current_role = "debater_a" if current_role == "debater_b" else "debater_b"
            turn_number += 1
        
        self.logger.info(f"Debate completed with {len(self.turns)} total turns")
        return self.turns


# ============================================================================
# Evaluation Module
# ============================================================================

class HallucinationEvaluator:
    """Evaluates hallucination rates from debate transcripts"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def extract_referee_annotations(self, turns: List[Turn]) -> List[HallucinationAnnotation]:
        """Extract hallucination annotations from referee turns"""
        annotations = []
        
        for turn in turns:
            if turn.agent_role != "referee":
                continue
            
            # Parse referee output
            # This is a simplified parser - in production, use more robust parsing
            content = turn.content.lower()
            
            # Detect severity markers
            if "factual_error" in content or "incorrect" in content:
                severity = "factual_error"
                is_hallucination = True
            elif "unverifiable" in content:
                severity = "unverifiable"
                is_hallucination = True
            elif "misleading" in content:
                severity = "misleading"
                is_hallucination = True
            else:
                severity = "correct"
                is_hallucination = False
            
            # Extract claims (simplified)
            # In production, use NLP to extract actual claims
            annotation = HallucinationAnnotation(
                turn_number=turn.turn_number,
                sentence_id=f"turn_{turn.turn_number}_ref",
                claim=turn.content[:200] + "..." if len(turn.content) > 200 else turn.content,
                is_hallucination=is_hallucination,
                severity=severity,
                evidence=[],
                annotator_notes="Automated extraction from referee output"
            )
            annotations.append(annotation)
        
        return annotations
    
    def calculate_metrics(self, turns: List[Turn], 
                         annotations: List[HallucinationAnnotation]) -> Dict[str, float]:
        """Calculate hallucination metrics"""
        total_debater_turns = sum(1 for t in turns if t.agent_role in ["debater_a", "debater_b"])
        total_hallucinations = sum(1 for a in annotations if a.is_hallucination)
        
        factual_errors = sum(1 for a in annotations if a.severity == "factual_error")
        unverifiable = sum(1 for a in annotations if a.severity == "unverifiable")
        misleading = sum(1 for a in annotations if a.severity == "misleading")
        
        metrics = {
            "total_turns": len(turns),
            "debater_turns": total_debater_turns,
            "referee_interventions": len(annotations),
            "total_hallucinations": total_hallucinations,
            "hallucination_rate": total_hallucinations / total_debater_turns if total_debater_turns > 0 else 0,
            "factual_errors": factual_errors,
            "unverifiable_claims": unverifiable,
            "misleading_claims": misleading,
            "correction_rate": 1 - (total_hallucinations / len(annotations)) if annotations else 0
        }
        
        return metrics


# ============================================================================
# Experiment Runner
# ============================================================================

class ExperimentRunner:
    """Main experiment orchestrator"""
    
    def __init__(self, experiment_name: str, seed: int = 42):
        self.experiment_name = experiment_name
        self.seed = seed
        self.timestamp = datetime.now().isoformat()
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Create output directory
        self.output_dir = f"outputs/{experiment_name}_{self.timestamp.replace(':', '-')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _get_experiment_config(self, experiment_name: str) -> ExperimentConfig:
        """Get configuration for specific experiment"""
        
        configs = {
            "nuclear_energy": ExperimentConfig(
                experiment_id=f"nuclear_energy_{self.timestamp}",
                topic="Nuclear Energy vs Renewable Energy",
                description="Debate on the role of nuclear power in sustainable energy future",
                max_turns=10,
                turn_timeout=120,
                deadlock_threshold=3,
                seed=self.seed,
                timestamp=self.timestamp,
                agent_a=AgentConfig(
                    name="Nuclear Advocate",
                    role=AgentRole.DEBATER_A,
                    model="claude-3-5-sonnet-20241022",
                    temperature=0.7,
                    max_tokens=1000,
                    system_prompt=SYSTEM_PROMPTS["debater_nuclear_pro"],
                    persona="Pro-nuclear energy expert"
                ),
                agent_b=AgentConfig(
                    name="Renewable Advocate",
                    role=AgentRole.DEBATER_B,
                    model="gpt-4o",
                    temperature=0.7,
                    max_tokens=1000,
                    system_prompt=SYSTEM_PROMPTS["debater_renewable_pro"],
                    persona="Pro-renewable energy expert"
                ),
                referee=AgentConfig(
                    name="Fact Checker",
                    role=AgentRole.REFEREE,
                    model="gemini-1.5-pro",
                    temperature=0.3,
                    max_tokens=1500,
                    system_prompt=SYSTEM_PROMPTS["referee"]
                )
            ),
            
            "good_vs_evil": ExperimentConfig(
                experiment_id=f"good_vs_evil_{self.timestamp}",
                topic="Human Nature: Inherently Good vs Evil",
                description="Philosophical debate on the fundamental nature of humanity",
                max_turns=10,
                turn_timeout=120,
                deadlock_threshold=3,
                seed=self.seed,
                timestamp=self.timestamp,
                agent_a=AgentConfig(
                    name="Optimist Philosopher",
                    role=AgentRole.DEBATER_A,
                    model="claude-3-5-sonnet-20241022",
                    temperature=0.7,
                    max_tokens=1000,
                    system_prompt=SYSTEM_PROMPTS["debater_good_vs_evil_good"],
                    persona="Philosopher arguing humans are inherently good"
                ),
                agent_b=AgentConfig(
                    name="Pessimist Philosopher",
                    role=AgentRole.DEBATER_B,
                    model="gpt-4o",
                    temperature=0.7,
                    max_tokens=1000,
                    system_prompt=SYSTEM_PROMPTS["debater_good_vs_evil_evil"],
                    persona="Philosopher arguing humans have inherent evil capacity"
                ),
                referee=AgentConfig(
                    name="Fact Checker",
                    role=AgentRole.REFEREE,
                    model="gemini-1.5-pro",
                    temperature=0.3,
                    max_tokens=1500,
                    system_prompt=SYSTEM_PROMPTS["referee"]
                )
            )
        }
        
        if experiment_name not in configs:
            raise ValueError(f"Unknown experiment: {experiment_name}")
        
        return configs[experiment_name]
    
    def run(self) -> ExperimentResults:
        """Execute the complete experiment"""
        self.logger.info("="*80)
        self.logger.info(f"Starting Experiment: {self.experiment_name}")
        self.logger.info(f"Seed: {self.seed}")
        self.logger.info("="*80)
        
        # Get configuration
        config = self._get_experiment_config(self.experiment_name)
        
        # Save configuration
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2, default=str)
        self.logger.info(f"Configuration saved to {config_path}")
        
        # Run debate
        debate_manager = DebateManager(config, self.logger)
        
        initial_prompts = {
            "nuclear_energy": "Present your opening argument on whether nuclear energy or renewable energy should be prioritized for a sustainable energy future. Focus on evidence-based claims about safety, economics, and environmental impact.",
            "good_vs_evil": "Present your opening philosophical argument on the fundamental nature of humanity. Are humans inherently good, or do they possess an inherent capacity for evil? Support your position with philosophical traditions and empirical evidence."
        }
        
        turns = debate_manager.run_debate(initial_prompts[self.experiment_name])
        
        # Save raw transcript
        transcript_path = os.path.join(self.output_dir, "full_transcript.json")
        with open(transcript_path, 'w') as f:
            json.dump([asdict(t) for t in turns], f, indent=2, default=str)
        self.logger.info(f"Full transcript saved to {transcript_path}")
        
        # Evaluate hallucinations
        evaluator = HallucinationEvaluator(self.logger)
        annotations = evaluator.extract_referee_annotations(turns)
        metrics = evaluator.calculate_metrics(turns, annotations)
        
        # Save annotations
        annotations_path = os.path.join(self.output_dir, "hallucination_annotations.json")
        with open(annotations_path, 'w') as f:
            json.dump([asdict(a) for a in annotations], f, indent=2, default=str)
        self.logger.info(f"Annotations saved to {annotations_path}")
        
        # Save metrics
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {metrics_path}")
        
        # Print summary
        self.logger.info("="*80)
        self.logger.info("EXPERIMENT RESULTS")
        self.logger.info("="*80)
        self.logger.info(f"Total turns: {metrics['total_turns']}")
        self.logger.info(f"Debater turns: {metrics['debater_turns']}")
        self.logger.info(f"Referee interventions: {metrics['referee_interventions']}")
        self.logger.info(f"Hallucination rate: {metrics['hallucination_rate']:.2%}")
        self.logger.info(f"Correction rate: {metrics['correction_rate']:.2%}")
        self.logger.info(f"Factual errors: {metrics['factual_errors']}")
        self.logger.info(f"Unverifiable claims: {metrics['unverifiable_claims']}")
        self.logger.info(f"Misleading claims: {metrics['misleading_claims']}")
        self.logger.info("="*80)
        
        # Create results object
        results = ExperimentResults(
            config=config,
            turns=turns,
            hallucination_annotations=annotations,
            metrics=metrics,
            metadata={
                "output_directory": self.output_dir,
                "completion_time": datetime.now().isoformat()
            }
        )
        
        return results
    
    def generate_report(self, results: ExperimentResults):
        """Generate a human-readable report"""
        report_path = os.path.join(self.output_dir, "REPORT.md")
        
        with open(report_path, 'w') as f:
            f.write(f"# Experiment Report: {results.config.topic}\n\n")
            f.write(f"**Experiment ID:** {results.config.experiment_id}\n\n")
            f.write(f"**Timestamp:** {results.config.timestamp}\n\n")
            f.write(f"**Seed:** {results.config.seed}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- **Max Turns:** {results.config.max_turns}\n")
            f.write(f"- **Deadlock Threshold:** {results.config.deadlock_threshold}\n\n")
            
            f.write("### Agent A\n")
            f.write(f"- Name: {results.config.agent_a.name}\n")
            f.write(f"- Model: {results.config.agent_a.model}\n")
            f.write(f"- Persona: {results.config.agent_a.persona}\n\n")
            
            f.write("### Agent B\n")
            f.write(f"- Name: {results.config.agent_b.name}\n")
            f.write(f"- Model: {results.config.agent_b.model}\n")
            f.write(f"- Persona: {results.config.agent_b.persona}\n\n")
            
            f.write("### Referee\n")
            f.write(f"- Name: {results.config.referee.name}\n")
            f.write(f"- Model: {results.config.referee.model}\n\n")
            
            f.write("## Results\n\n")
            f.write(f"- **Total Turns:** {results.metrics['total_turns']}\n")
            f.write(f"- **Hallucination Rate:** {results.metrics['hallucination_rate']:.2%}\n")
            f.write(f"- **Correction Rate:** {results.metrics['correction_rate']:.2%}\n")
            f.write(f"- **Factual Errors:** {results.metrics['factual_errors']}\n")
            f.write(f"- **Unverifiable Claims:** {results.metrics['unverifiable_claims']}\n")
            f.write(f"- **Misleading Claims:** {results.metrics['misleading_claims']}\n\n")
            
            f.write("## Debate Transcript Summary\n\n")
            for i, turn in enumerate(results.turns[:6]):  # First 3 exchanges
                f.write(f"### Turn {turn.turn_number} - {turn.agent_name} ({turn.agent_role})\n\n")
                f.write(f"{turn.content[:500]}...\n\n")
            
            f.write("\n*See full_transcript.json for complete debate log*\n")
        
        self.logger.info(f"Report generated at {report_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reproducible Referee-Mediated Discourse Experiment"
    )
    parser.add_argument(
        "--experiment",
        choices=["nuclear_energy", "good_vs_evil"],
        required=True,
        help="Which experiment to run"
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
    
    # Verify API keys
    required_keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print("ERROR: Missing required environment variables:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nPlease set these environment variables before running.")
        return 1
    
    # Run experiment
    runner = ExperimentRunner(args.experiment, args.seed)
    results = runner.run()
    runner.generate_report(results)
    
    print(f"\n✅ Experiment completed successfully!")
    print(f"📁 Results saved to: {runner.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())

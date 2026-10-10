import os
import time
from typing import Any, Dict, Optional, Tuple
from laya import Router


DEFAULT_QUESTIONS = {
    "destination": {
        "type": "choice",
        "instructions": "Which specialist should handle this request?",
        "criteria": {
            "whatsapp": "WhatsApp messages, chat support, notifications, direct messaging, business communication",
            "instagram": "Instagram posts, stories, reels, comments, DMs, social media engagement and marketing",
            "websearch": "Live web search, internet research, online documentation lookup, public web articles",
        },
    },
    "urgent": {
        "type": "choice",
        "instructions": "How urgent is this request?",
        "criteria": {
            "high": "Time-sensitive message, breaking news update, immediate delivery required, high priority notification",
            "normal": "Routine inquiry, scheduled social post, background lookup or standard message",
        },
    },
}

INTENT_MAP = {
    "whatsapp": "whatsapp_messaging",
    "instagram": "instagram_engagement",
    "websearch": "web_research",
}


class LayaSupervisor:
    """Supervisor agent that uses Laya's System 1 non-autoregressive decision engine

    and Neo4j graph episodic memory to route user requests with microsecond latency.
    """

    _instance: Optional["LayaSupervisor"] = None

    def __init__(self, device: str = "mps"):
        try:
            self.router = Router(device=device)
        except Exception:
            # Fallback to cpu if mps or custom device fails
            self.router = Router(device="cpu")

        # Warm up the model once
        self._warmup()

    def _warmup(self):
        """Warm up the router to avoid cold-start latency in subsequent requests."""
        dummy_state = "Search for information online."
        try:
            self.router.predict(dummy_state, DEFAULT_QUESTIONS)
        except Exception:
            pass

    @classmethod
    def get_instance(cls, device: str = "mps") -> "LayaSupervisor":
        """Get or create singleton supervisor instance."""
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance

    def predict_route(
        self,
        query: str,
        questions: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """Run Laya router model inference and time the operation.

        Returns (result_dict, elapsed_ms).
        """
        q = questions or DEFAULT_QUESTIONS
        start = time.perf_counter()
        raw_result = self.router.predict(query, q)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return raw_result, round(elapsed_ms, 2)

    def evaluate_decision(
        self,
        laya_result: Dict[str, Any],
        memory_context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float]:
        """Synthesize Laya prediction with Neo4j episodic memory to understand

        and justify the final routing decision.
        Returns (decision_dict, elapsed_ms).
        """
        start = time.perf_counter()

        answers = laya_result.get("answers", {})
        dest_answer = answers.get("destination", {})
        urgent_answer = answers.get("urgent", {})

        pred_destination = dest_answer.get("choice", "websearch")
        pred_confidence = float(dest_answer.get("answer_confidence", 0.5))
        probabilities = dest_answer.get("probabilities", {})

        pred_urgency = urgent_answer.get("choice", "normal")
        urgency_confidence = float(urgent_answer.get("answer_confidence", 0.5))

        # Analyze Neo4j memory context
        matched_intents = memory_context.get("matched_intents", [])
        matched_episodes = memory_context.get("matched_episodes", [])

        # Count specialist recommendations from Neo4j episodes
        episode_votes: Dict[str, int] = {}
        for ep in matched_episodes:
            sel = ep.get("selected")
            if sel:
                episode_votes[sel] = episode_votes.get(sel, 0) + ep.get("relevance_score", 1)

        best_memory_specialist = max(episode_votes, key=episode_votes.get) if episode_votes else None

        final_destination = pred_destination
        final_confidence = pred_confidence
        memory_agreement = False
        decision_source = "Laya Neural Router"
        rationale_items = []

        if best_memory_specialist and best_memory_specialist == pred_destination:
            # Full agreement between Laya and Neo4j Memory
            memory_agreement = True
            decision_source = "Consensus (Laya + Neo4j Memory)"
            boosted = min(0.99, pred_confidence * 1.10)
            final_confidence = round(boosted, 4)
            corroborating_count = len([e for e in matched_episodes if e.get("selected") == pred_destination])
            rationale_items.append(
                f"Consensus reached: Laya neural model ({pred_confidence*100:.1f}%) and Neo4j episodic memory agree on '{pred_destination}'."
            )
            rationale_items.append(
                f"Neo4j corroborates via {corroborating_count} verified historical episode(s) under intent '{INTENT_MAP.get(pred_destination, 'web_research')}'."
            )
        elif best_memory_specialist and pred_confidence < 0.45 and episode_votes[best_memory_specialist] >= 2:
            # Memory override/rescue: Laya was uncertain but Neo4j has strong historical precedent
            final_destination = best_memory_specialist
            final_confidence = 0.85
            decision_source = "Neo4j Memory Override"
            memory_agreement = False
            top_ep = next((e for e in matched_episodes if e.get("selected") == best_memory_specialist), {})
            rationale_items.append(
                f"Neo4j episodic memory guided decision to '{final_destination}' (historical precedent: '{top_ep.get('text', '')[:50]}...')."
            )
            rationale_items.append(
                f"Laya neural model was uncertain (confidence {pred_confidence*100:.1f}%), so Neo4j institutional memory provided the decision."
            )
        elif matched_episodes:
            # Laya had clear confidence, though some past queries in memory were slightly different
            decision_source = "Laya Neural Prior"
            rationale_items.append(
                f"Laya neural router prioritized '{pred_destination}' with {pred_confidence*100:.1f}% confidence ({probabilities.get(pred_destination, 0)*100:.1f}% probability)."
            )
            rationale_items.append(
                f"Neo4j memory contained {len(matched_episodes)} related episode(s), evaluated alongside model criteria."
            )
        else:
            decision_source = "Laya Zero-Shot (Novel Route)"
            rationale_items.append(
                f"Laya neural router classified this novel request to '{pred_destination}' with {pred_confidence*100:.1f}% confidence."
            )
            rationale_items.append("No prior episodes found in Neo4j; this execution will establish a new memory episode.")

        # Urgency rationale
        rationale_items.append(
            f"Triage urgency evaluated as '{pred_urgency.upper()}' ({urgency_confidence*100:.1f}% confidence)."
        )

        matched_intent_name = INTENT_MAP.get(final_destination, "web_research")
        full_rationale = " ".join(rationale_items)

        decision = {
            "destination": final_destination,
            "urgency": pred_urgency,
            "raw_confidence": round(pred_confidence, 4),
            "final_confidence": round(final_confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "matched_intent": matched_intent_name,
            "memory_agreement": memory_agreement,
            "decision_source": decision_source,
            "rationale": full_rationale,
            "memory_top_specialist": best_memory_specialist,
            "usage": laya_result.get("usage", {}),
        }

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return decision, round(elapsed_ms, 2)

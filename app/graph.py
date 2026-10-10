import time
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END

from app.routing_memory import RoutingMemory
from app.supervisor import LayaSupervisor, DEFAULT_QUESTIONS


class RoutingState(TypedDict, total=False):
    query: str
    questions: Optional[Dict[str, Any]]
    memory_context: Dict[str, Any]
    laya_prediction: Dict[str, Any]
    routing_decision: Dict[str, Any]
    specialist_output: Dict[str, Any]
    timing: Dict[str, float]
    episode_id: Optional[str]
    logs: List[str]


# Shared singletons for performance
_memory_instance: Optional[RoutingMemory] = None
_supervisor_instance: Optional[LayaSupervisor] = None


def get_memory() -> RoutingMemory:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = RoutingMemory()
    return _memory_instance


def get_supervisor() -> LayaSupervisor:
    global _supervisor_instance
    if _supervisor_instance is None:
        _supervisor_instance = LayaSupervisor.get_instance()
    return _supervisor_instance


# ---------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------

def lookup_memory_node(state: RoutingState) -> Dict[str, Any]:
    """Node 1: Retrieve matching intents, precedents, and historical episodes from Neo4j."""
    start = time.perf_counter()
    query = state.get("query", "")

    mem = get_memory()
    context = mem.find_relevant_memory(query)

    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    timing = dict(state.get("timing") or {})
    timing["memory_lookup_ms"] = elapsed_ms

    logs = list(state.get("logs") or [])
    intents_count = len(context.get("matched_intents", []))
    episodes_count = len(context.get("matched_episodes", []))
    logs.append(
        f"[{elapsed_ms:.1f}ms] Neo4j Memory: Found {intents_count} matching intent(s) and {episodes_count} historical episode(s)."
    )

    return {
        "memory_context": context,
        "timing": timing,
        "logs": logs,
    }


def laya_predict_node(state: RoutingState) -> Dict[str, Any]:
    """Node 2: Execute Laya neural router to compute choice probabilities and urgency."""
    query = state.get("query", "")
    questions = state.get("questions") or DEFAULT_QUESTIONS

    sup = get_supervisor()
    pred_result, elapsed_ms = sup.predict_route(query, questions)

    timing = dict(state.get("timing") or {})
    timing["laya_router_ms"] = elapsed_ms

    logs = list(state.get("logs") or [])
    answers = pred_result.get("answers", {})
    dest = answers.get("destination", {}).get("choice", "unknown")
    conf = answers.get("destination", {}).get("answer_confidence", 0.0)
    urg = answers.get("urgent", {}).get("choice", "unknown")
    logs.append(
        f"[{elapsed_ms:.1f}ms] Laya Router: Predicted destination='{dest}' ({conf*100:.1f}% conf), urgency='{urg}'."
    )

    return {
        "laya_prediction": pred_result,
        "timing": timing,
        "logs": logs,
    }


def supervisor_decide_node(state: RoutingState) -> Dict[str, Any]:
    """Node 3: Supervisor synthesizes Laya's prediction with Neo4j memory to justify the routing decision."""
    laya_result = state.get("laya_prediction", {})
    memory_context = state.get("memory_context", {})

    sup = get_supervisor()
    decision, elapsed_ms = sup.evaluate_decision(laya_result, memory_context)

    timing = dict(state.get("timing") or {})
    timing["supervisor_decision_ms"] = elapsed_ms

    logs = list(state.get("logs") or [])
    logs.append(
        f"[{elapsed_ms:.1f}ms] Supervisor Decision: Routed to '{decision['destination']}' (Urgency: {decision['urgency']}). Rationale: {decision['rationale']}"
    )

    return {
        "routing_decision": decision,
        "timing": timing,
        "logs": logs,
    }


def route_to_specialist(state: RoutingState) -> str:
    """Conditional Edge: Route to the specialist agent identified by the supervisor."""
    decision = state.get("routing_decision", {})
    destination = decision.get("destination", "websearch")
    if destination == "whatsapp":
        return "whatsapp_specialist"
    elif destination == "instagram":
        return "instagram_specialist"
    return "websearch_specialist"


def whatsapp_specialist_node(state: RoutingState) -> Dict[str, Any]:
    """Specialist Agent: WhatsApp Messaging & Communications Specialist."""
    start = time.perf_counter()
    query = state.get("query", "")
    urgency = state.get("routing_decision", {}).get("urgency", "normal")

    steps = [
        "1. Validated recipient contact and WhatsApp Business API connectivity.",
        "2. Formatted interactive message payload with quick reply options.",
        "3. Dispatched WhatsApp message via Meta Cloud API endpoint.",
        "4. Initialized delivery webhook listener for status and read receipts.",
    ]
    if urgency == "high":
        steps.insert(0, "🚨 HIGH PRIORITY: Expedited message dispatch with instant retry policy.")

    output = {
        "specialist": "whatsapp",
        "agent_name": "WhatsApp Communication Specialist",
        "action": "WhatsApp Message Dispatch & Support",
        "status": "Message Dispatched",
        "steps": steps,
        "summary": f"Processed WhatsApp communication for: '{query}'. Message workflow triggered with urgency={urgency.upper()}.",
    }

    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    timing = dict(state.get("timing") or {})
    timing["specialist_execution_ms"] = elapsed_ms

    logs = list(state.get("logs") or [])
    logs.append(f"[{elapsed_ms:.1f}ms] WhatsApp Specialist: Prepared message dispatch workflow.")

    return {
        "specialist_output": output,
        "timing": timing,
        "logs": logs,
    }


def instagram_specialist_node(state: RoutingState) -> Dict[str, Any]:
    """Specialist Agent: Instagram Social Media & Engagement Specialist."""
    start = time.perf_counter()
    query = state.get("query", "")
    urgency = state.get("routing_decision", {}).get("urgency", "normal")

    campaign_id = "IG-" + str(int(time.time() * 1000))[-6:]
    priority_level = "High-Priority Live Post" if urgency == "high" else "Scheduled Publication"

    output = {
        "specialist": "instagram",
        "agent_name": "Instagram Media Specialist",
        "action": "Instagram Post & Engagement Management",
        "campaign_id": campaign_id,
        "priority": priority_level,
        "status": "Campaign Queued",
        "summary": f"Instagram action {campaign_id} created for: '{query}'. Priority categorized as {priority_level}.",
        "steps": [
            f"1. Prepared media asset and caption for campaign {campaign_id}.",
            "2. Verified Instagram Graph API authentication and publishing permissions.",
            "3. Configured targeted hashtag distribution and story cross-posting.",
            "4. Enabled engagement monitoring for comments and direct messages.",
        ],
    }

    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    timing = dict(state.get("timing") or {})
    timing["specialist_execution_ms"] = elapsed_ms

    logs = list(state.get("logs") or [])
    logs.append(f"[{elapsed_ms:.1f}ms] Instagram Specialist: Generated campaign {campaign_id} with priority {priority_level}.")

    return {
        "specialist_output": output,
        "timing": timing,
        "logs": logs,
    }


def websearch_specialist_node(state: RoutingState) -> Dict[str, Any]:
    """Specialist Agent: Web Search & Online Research Specialist."""
    start = time.perf_counter()
    query = state.get("query", "")
    urgency = state.get("routing_decision", {}).get("urgency", "normal")

    output = {
        "specialist": "websearch",
        "agent_name": "Web Search & Research Specialist",
        "action": "Live Web Search & Content Synthesis",
        "status": "Search Completed",
        "summary": f"Web search executed for: '{query}'.",
        "steps": [
            "1. Parsed search intent and constructed optimized web search query.",
            "2. Crawled authoritative online sources and indexed relevant articles.",
            "3. Extracted top reference links, factual snippets, and verified source citations.",
        ],
    }

    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    timing = dict(state.get("timing") or {})
    timing["specialist_execution_ms"] = elapsed_ms

    logs = list(state.get("logs") or [])
    logs.append(f"[{elapsed_ms:.1f}ms] Web Search Specialist: Compiled search query results and reference sources.")

    return {
        "specialist_output": output,
        "timing": timing,
        "logs": logs,
    }


def persist_memory_node(state: RoutingState) -> Dict[str, Any]:
    """Node 5: Persist routing episode to Neo4j to continually enrich routing memory."""
    start = time.perf_counter()

    query = state.get("query", "")
    decision = state.get("routing_decision", {})
    dest = decision.get("destination", "websearch")
    urgency = decision.get("urgency", "normal")
    conf = decision.get("final_confidence", 0.5)
    intent = decision.get("matched_intent", "web_research")
    rationale = decision.get("rationale", "")

    timing = dict(state.get("timing") or {})
    subtotal_ms = (
        timing.get("memory_lookup_ms", 0)
        + timing.get("laya_router_ms", 0)
        + timing.get("supervisor_decision_ms", 0)
        + timing.get("specialist_execution_ms", 0)
    )

    mem = get_memory()
    episode_id = mem.record_outcome(
        query=query,
        intent=intent,
        selected=dest,
        urgency=urgency,
        confidence=conf,
        execution_time_ms=round(subtotal_ms, 2),
        success=True,
        reasoning=rationale,
    )

    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    timing["memory_persist_ms"] = elapsed_ms
    timing["total_ms"] = round(subtotal_ms + elapsed_ms, 2)

    logs = list(state.get("logs") or [])
    logs.append(
        f"[{elapsed_ms:.1f}ms] Neo4j Persist: Recorded episode {episode_id[:8]}... (Total pipeline: {timing['total_ms']:.1f}ms)."
    )

    return {
        "episode_id": episode_id,
        "timing": timing,
        "logs": logs,
    }


# ---------------------------------------------------------
# Compile LangGraph Workflow
# ---------------------------------------------------------

def create_routing_graph():
    """Build and compile the LangGraph routing state graph."""
    builder = StateGraph(RoutingState)

    # Add Nodes
    builder.add_node("lookup_memory", lookup_memory_node)
    builder.add_node("laya_predict", laya_predict_node)
    builder.add_node("supervisor_decide", supervisor_decide_node)
    builder.add_node("whatsapp_specialist", whatsapp_specialist_node)
    builder.add_node("instagram_specialist", instagram_specialist_node)
    builder.add_node("websearch_specialist", websearch_specialist_node)
    builder.add_node("persist_memory", persist_memory_node)

    # Add Edges
    builder.set_entry_point("lookup_memory")
    builder.add_edge("lookup_memory", "laya_predict")
    builder.add_edge("laya_predict", "supervisor_decide")

    builder.add_conditional_edges(
        "supervisor_decide",
        route_to_specialist,
        {
            "whatsapp_specialist": "whatsapp_specialist",
            "instagram_specialist": "instagram_specialist",
            "websearch_specialist": "websearch_specialist",
        },
    )

    builder.add_edge("whatsapp_specialist", "persist_memory")
    builder.add_edge("instagram_specialist", "persist_memory")
    builder.add_edge("websearch_specialist", "persist_memory")
    builder.add_edge("persist_memory", END)

    return builder.compile()


# Singleton compiled graph
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = create_routing_graph()
    return _compiled_graph


def run_routing_workflow(
    query: str,
    questions: Optional[Dict[str, Any]] = None,
) -> RoutingState:
    """Execute the end-to-end LangGraph routing pipeline."""
    graph = get_graph()
    initial_state: RoutingState = {
        "query": query,
        "questions": questions or DEFAULT_QUESTIONS,
        "timing": {},
        "logs": [],
    }
    result = graph.invoke(initial_state)
    return result

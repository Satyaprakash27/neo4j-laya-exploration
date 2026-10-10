"""CLI demonstration of Laya + LangGraph Agent Router with Neo4j Episodic Memory.

Similar to Laya/main.py, but orchestrated via LangGraph, backed by Neo4j institutional memory,
and timing every single phase of execution.
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.routing_memory import RoutingMemory
from app.graph import run_routing_workflow


def main():
    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]⚡ Laya + LangGraph Agent Router with Neo4j Memory[/bold cyan]\n"
            "[dim]Institutional Graph Memory + Non-Autoregressive System 1 Routing with Sub-Millisecond Timing[/dim]",
            border_style="cyan",
        )
    )

    # 1. Verify / Seed Neo4j Memory
    console.print("[bold yellow]1. Connecting to Neo4j Memory...[/bold yellow]")
    t0 = time.perf_counter()
    mem = RoutingMemory()
    mem.seed()
    t_seed = (time.perf_counter() - t0) * 1000.0
    stats = mem.get_stats()
    console.print(
        f"[green]✓ Connected & Seeded Neo4j in {t_seed:.1f}ms. Total episodes: {stats['episodes_count']}, Known intents: {stats['intents_count']}[/green]\n"
    )

    # 2. Sample Request (identical to Laya/main.py)
    query = """Send a WhatsApp message to customer +14155552671
confirming their reservation details and asking for quick confirmation."""

    console.print(Panel(query.strip(), title="[bold white]Incoming Request / State[/bold white]", border_style="blue"))

    # 3. Execute LangGraph Workflow
    console.print("\n[bold yellow]2. Executing LangGraph Routing Workflow...[/bold yellow]")
    result = run_routing_workflow(query)

    decision = result["routing_decision"]
    specialist = result["specialist_output"]
    timing = result["timing"]
    logs = result["logs"]

    # 4. Timing Summary Table
    table = Table(title="⏱️ Phase Execution Timings (Milliseconds)", border_style="magenta")
    table.add_column("Pipeline Stage", style="bold cyan")
    table.add_column("Latency (ms)", justify="right", style="bold green")
    table.add_column("Description", style="dim")

    table.add_row("1. Neo4j Memory Lookup", f"{timing.get('memory_lookup_ms', 0):.2f} ms", "Cypher query for matching intents & historical episodes")
    table.add_row("2. Laya Router Predict", f"{timing.get('laya_router_ms', 0):.2f} ms", "Neural System 1 non-autoregressive classification")
    table.add_row("3. Supervisor Synthesis", f"{timing.get('supervisor_decision_ms', 0):.2f} ms", "Synthesized Laya probabilities + Neo4j memory consensus")
    table.add_row("4. Specialist Execution", f"{timing.get('specialist_execution_ms', 0):.2f} ms", f"Routed to {decision['destination'].upper()} specialist agent")
    table.add_row("5. Neo4j Memory Persist", f"{timing.get('memory_persist_ms', 0):.2f} ms", "Persisted new episode & outcome into Neo4j graph")
    table.add_section()
    table.add_row("[bold white]Total Workflow Latency[/bold white]", f"[bold yellow]{timing.get('total_ms', 0):.2f} ms[/bold yellow]", "End-to-end LangGraph state machine time")

    console.print(table)

    # 5. Routing Decision & Explanation
    console.print("\n[bold yellow]3. Routing Decision & Memory Rationale:[/bold yellow]")
    urgency_color = "bold red" if decision["urgency"] == "high" else "bold cyan"
    decision_panel = (
        f"[bold]Selected Destination:[/bold] [bold green]{decision['destination'].upper()}[/bold green]\n"
        f"[bold]Assessed Urgency:[/bold] [{urgency_color}]{decision['urgency'].upper()}[/{urgency_color}]\n"
        f"[bold]Calibrated Confidence:[/bold] {decision['final_confidence']*100:.1f}%\n"
        f"[bold]Decision Source:[/bold] {decision.get('decision_source')}\n"
        f"[bold]Specialist Probabilities:[/bold] {decision['probabilities']}\n\n"
        f"[bold]Synthesis Rationale:[/bold]\n{decision['rationale']}"
    )
    console.print(Panel(decision_panel, title="[bold green]Routing Decision[/bold green]", border_style="green"))

    # 6. Specialist Output
    console.print("\n[bold yellow]4. Specialist Output Action:[/bold yellow]")
    steps_txt = "\n".join([f"  • {s}" for s in specialist.get("steps", [])])
    specialist_panel = (
        f"[bold]Specialist Agent:[/bold] {specialist['agent_name']}\n"
        f"[bold]Action:[/bold] {specialist['action']}\n"
        f"[bold]Summary:[/bold] {specialist['summary']}\n\n"
        f"[bold]Remediation Workflow:[/bold]\n{steps_txt}"
    )
    console.print(Panel(specialist_panel, title="[bold magenta]Specialist Action Plan[/bold magenta]", border_style="magenta"))

    # 7. Workflow Trace Logs
    console.print("\n[bold yellow]5. LangGraph Execution Trace Logs:[/bold yellow]")
    for line in logs:
        console.print(f"  [dim cyan]>[/dim cyan] {line}")

    console.print(f"\n[green]✓ Episode recorded in Neo4j with ID: {result.get('episode_id')}[/green]")
    console.print("[dim]Run the Streamlit UI with: streamlit run app/streamlit_app.py[/dim]\n")


if __name__ == "__main__":
    main()

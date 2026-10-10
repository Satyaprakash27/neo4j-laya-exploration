import os
import sys
import time
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.routing_memory import RoutingMemory
from app.supervisor import DEFAULT_QUESTIONS, LayaSupervisor
from app.graph import get_graph, get_memory, run_routing_workflow


# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Laya + LangGraph Agent Router with Neo4j Memory",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .badge-urgent-high {
        background-color: #dc2626;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-urgent-normal {
        background-color: #0284c7;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-dest {
        background-color: #7c3aed;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Neo4j Memory Helper
# -----------------------------------------------------------------------------
def get_safe_memory():
    try:
        mem = get_memory()
        return mem, None
    except Exception as e:
        return None, str(e)


# -----------------------------------------------------------------------------
# Sidebar: Memory & System Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("⚡ System & Memory Status")

    mem, mem_err = get_safe_memory()
    if mem_err:
        st.error(f"Neo4j Connection Failed: {mem_err}")
    else:
        st.success("🟢 Neo4j Database Connected")
        stats = mem.get_stats()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Episodes", stats.get("episodes_count", 0))
            st.metric("Known Intents", stats.get("intents_count", 0))
        with c2:
            st.metric("Specialists", stats.get("specialists_count", 0))
            st.metric("Success Rate", f"{stats.get('success_rate', 0)}%")

        st.markdown("---")
        st.markdown("### 🗄️ Memory Management")

        col_seed, col_refresh = st.columns(2)
        with col_seed:
            if st.button("🌱 Seed Memory", use_container_width=True):
                with st.spinner("Seeding Neo4j..."):
                    mem.seed()
                    st.toast("Seeded core IT routes and episodes into Neo4j!", icon="✅")
                    time.sleep(0.5)
                    st.rerun()

        with col_refresh:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        if st.button("⚠️ Clear Memory", use_container_width=True):
            mem.clear_memory()
            st.toast("Cleared Neo4j routing memory.", icon="🧹")
            time.sleep(0.5)
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Laya Router Engine")
    device_opt = st.selectbox("Inference Device", ["mps", "cpu"], index=0)
    st.caption("Laya is a non-autoregressive calibrated decision engine designed for sub-100ms routing.")


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">⚡ Laya + LangGraph Agent Router</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">LangGraph multi-agent routing powered by Laya\'s non-autoregressive decision model and Neo4j institutional memory. Every step is timed with microsecond precision.</div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Quick Request Presets
# -----------------------------------------------------------------------------
st.markdown("##### 📌 Select a Preset Scenario or Type a Request:")

preset_cols = st.columns(4)

preset_1 = """Send an urgent WhatsApp message to customer +14155552671
with the flight booking confirmation and check-in link."""

preset_2 = """Post our new summer collection product showcase video on Instagram
with trending fashion hashtags and an interactive story poll."""

preset_3 = """Check the direct messages and comments on our latest Instagram reel
and reply to product availability questions."""

preset_4 = """Search the web for the latest advancements and benchmark results
in quantum computing hardware architectures."""

selected_preset = None

with preset_cols[0]:
    if st.button("💬 WhatsApp: Client Confirmation", use_container_width=True):
        st.session_state["query_input"] = preset_1

with preset_cols[1]:
    if st.button("📸 Instagram: New Post & Story", use_container_width=True):
        st.session_state["query_input"] = preset_2

with preset_cols[2]:
    if st.button("📬 Instagram: DM & Comments", use_container_width=True):
        st.session_state["query_input"] = preset_3

with preset_cols[3]:
    if st.button("🌐 Web Search: Tech Benchmarks", use_container_width=True):
        st.session_state["query_input"] = preset_4

default_text = st.session_state.get("query_input", preset_1)
user_query = st.text_area(
    "User Request / State:",
    value=default_text,
    height=110,
    placeholder="Describe the user's issue or service request...",
)

with st.expander("⚙️ Customize Routing Questions & Criteria (Laya Schema)", expanded=False):
    st.info("These criteria are fed directly into Laya's non-autoregressive classifier, matching the setup from Laya/main.py.")
    st.json(DEFAULT_QUESTIONS)

run_button = st.button("🚀 Execute LangGraph Routing Workflow", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Execution & Display
# -----------------------------------------------------------------------------
if run_button and user_query.strip():
    with st.spinner("Executing LangGraph routing pipeline..."):
        start_wall = time.perf_counter()
        result = run_routing_workflow(user_query)
        total_wall_ms = round((time.perf_counter() - start_wall) * 1000.0, 2)

    st.session_state["last_result"] = result
    st.session_state["last_query"] = user_query


if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    timing = result.get("timing", {})
    decision = result.get("routing_decision", {})
    spec_out = result.get("specialist_output", {})
    dest = decision.get("destination", "websearch")
    urgency = decision.get("urgency", "normal")
    final_conf = decision.get("final_confidence", 0.0)
    raw_conf = decision.get("raw_confidence", 0.0)
    probs = decision.get("probabilities", {})
    rationale = decision.get("rationale", "")
    decision_source = decision.get("decision_source", "Laya Neural Router")
    episode_id = result.get("episode_id")

    st.markdown("---")
    st.markdown("### ⏱️ Operation Timing Breakdown")

    # Timing KPI Metrics
    t1, t2, t3, t4, t5, t6 = st.columns(6)
    with t1:
        st.metric("1. Memory Lookup", f"{timing.get('memory_lookup_ms', 0):.1f} ms", help="Neo4j Cypher query for matching intents & historical episodes")
    with t2:
        st.metric("2. Laya Router", f"{timing.get('laya_router_ms', 0):.1f} ms", help="Laya neural classifier predicting specialist choice & urgency")
    with t3:
        st.metric("3. Supervisor", f"{timing.get('supervisor_decision_ms', 0):.2f} ms", help="Supervisor synthesizing Laya prediction + Neo4j episodic memory")
    with t4:
        st.metric("4. Specialist", f"{timing.get('specialist_execution_ms', 0):.2f} ms", help="Targeted specialist agent execution & action plan")
    with t5:
        st.metric("5. Neo4j Persist", f"{timing.get('memory_persist_ms', 0):.1f} ms", help="Writing executed routing episode and outcome into Neo4j graph")
    with t6:
        st.metric("Total Latency", f"{timing.get('total_ms', 0):.1f} ms", delta=None, help="End-to-end LangGraph state machine execution")

    # Main Decision Card
    st.markdown("### 🎯 Routing Decision & Memory Context")
    col_dec, col_spec = st.columns([1, 1])

    with col_dec:
        st.markdown(
            f"""
            <div style="background-color: #0f172a; padding: 18px; border-radius: 10px; border: 1px solid #1e293b;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div>
                        <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Selected Destination:</span>
                        <span class="badge-dest">{dest.upper()} SPECIALIST</span>
                    </div>
                    <div>
                        <span class="{'badge-urgent-high' if urgency == 'high' else 'badge-urgent-normal'}">URGENCY: {urgency.upper()}</span>
                    </div>
                </div>
                <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 8px;">
                    <b>Decision Source:</b> <span style="color: #38bdf8;">{decision_source}</span>
                </p>
                <p style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 14px;">
                    <b>Calibrated Confidence:</b> <span style="font-size: 1.1rem; color: #10b981; font-weight: 700;">{final_conf * 100:.1f}%</span> (Laya Raw: {raw_conf * 100:.1f}%)
                </p>
                <div style="background-color: #1e293b; padding: 12px; border-radius: 8px; border-left: 4px solid #6366f1;">
                    <b style="color: #e2e8f0;">Decision Rationale:</b><br/>
                    <span style="color: #cbd5e1; font-size: 0.9rem;">{rationale}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### 📊 Laya Specialist Probability Distribution:")
        for sp_name, prob_val in probs.items():
            st.progress(float(prob_val), text=f"{sp_name.capitalize()}: {prob_val * 100:.1f}%")

    with col_spec:
        st.markdown(
            f"""
            <div style="background-color: #0f172a; padding: 18px; border-radius: 10px; border: 1px solid #1e293b; height: 100%;">
                <div style="margin-bottom: 10px;">
                    <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">🛠️ {spec_out.get('agent_name', 'Specialist')}</span>
                </div>
                <p style="color: #38bdf8; font-size: 0.95rem; margin-bottom: 6px;"><b>Action:</b> {spec_out.get('action')}</p>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 12px;">{spec_out.get('summary')}</p>
                <div style="background-color: #1e293b; padding: 12px; border-radius: 8px;">
                    <b style="color: #e2e8f0;">Action & Remediation Steps:</b>
                    <ul style="color: #cbd5e1; font-size: 0.88rem; margin-top: 6px; padding-left: 20px;">
                        {''.join([f'<li>{s}</li>' for s in spec_out.get('steps', [])])}
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Episode Feedback Bar
    st.markdown("---")
    fb_col1, fb_col2 = st.columns([3, 1])
    with fb_col1:
        st.caption(f"Neo4j Episode Recorded: `{episode_id}` | Saved with execution time {timing.get('total_ms', 0):.1f} ms.")
    with fb_col2:
        fb_c1, fb_c2 = st.columns(2)
        with fb_c1:
            if st.button("👍 Correct", key="thumb_up"):
                if mem and episode_id:
                    mem.update_feedback(episode_id, success=True, feedback="Verified correct by user")
                    st.toast("Feedback recorded in Neo4j!", icon="✅")
        with fb_c2:
            if st.button("👎 Incorrect", key="thumb_down"):
                if mem and episode_id:
                    mem.update_feedback(episode_id, success=False, feedback="Marked incorrect by user")
                    st.toast("Flagged incorrect in Neo4j!", icon="⚠️")

    # -------------------------------------------------------------------------
    # Inspection Tabs
    # -------------------------------------------------------------------------
    st.markdown("### 🔍 Deep Inspection & Audit")
    tab_logs, tab_memory, tab_graph = st.tabs(["📋 Execution Trace & Logs", "🧠 Neo4j Memory Explorer", "🗺️ LangGraph Topology"])

    with tab_logs:
        st.markdown("##### Step-by-Step LangGraph Execution Trace:")
        for log in result.get("logs", []):
            st.code(log, language="bash")

    with tab_memory:
        mem_ctx = result.get("memory_context", {})
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.markdown("##### 📌 Relevant Precedents Found for this Query:")
            matched_eps = mem_ctx.get("matched_episodes", [])
            if matched_eps:
                st.dataframe(pd.DataFrame(matched_eps), use_container_width=True)
            else:
                st.info("No prior matching episodes found for this query in Neo4j.")

        with c_m2:
            st.markdown("##### 🌐 Known Intent Routes in Neo4j:")
            all_r = mem_ctx.get("all_routes", [])
            if all_r:
                st.dataframe(pd.DataFrame(all_r), use_container_width=True)
            else:
                st.info("No intent routes currently in Neo4j. Click 'Seed Memory' in the sidebar.")

        st.markdown("##### 📜 Recent Neo4j Routing Episodes (Live):")
        if mem:
            recent_eps = mem.get_recent_episodes(limit=15)
            if recent_eps:
                st.dataframe(pd.DataFrame(recent_eps), use_container_width=True)
            else:
                st.info("No episodes recorded yet.")

    with tab_graph:
        st.markdown("##### LangGraph Architecture & Conditional Dispatch:")
        st.markdown(
            """
            ```mermaid
            flowchart TD
                START([User Request / State]) --> LookupMemory[1. Neo4j Memory Lookup<br/><i>Retrieve Intents & Precedents</i>]
                LookupMemory --> LayaPredict[2. Laya Neural Router<br/><i>Fast Calibrated Choice & Urgency</i>]
                LayaPredict --> SupervisorDecide[3. Supervisor Synthesis<br/><i>Synthesize Prediction + Graph Memory</i>]
                
                SupervisorDecide -->|destination == 'whatsapp'| WhatsAppSpec[4a. WhatsApp Specialist<br/><i>Direct Messaging & Support</i>]
                SupervisorDecide -->|destination == 'instagram'| InstaSpec[4b. Instagram Specialist<br/><i>Media Posts & Engagement</i>]
                SupervisorDecide -->|destination == 'websearch'| WebSearchSpec[4c. Web Search Specialist<br/><i>Live Search & Research</i>]
                
                WhatsAppSpec --> PersistMemory[5. Neo4j Memory Persist<br/><i>Store Episode & Audit Trail</i>]
                InstaSpec --> PersistMemory
                WebSearchSpec --> PersistMemory
                
                PersistMemory --> END_NODE([Completed Response + Timings])
            ```
            """
        )

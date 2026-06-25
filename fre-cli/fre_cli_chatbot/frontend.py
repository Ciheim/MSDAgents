#!/usr/bin/env python3
"""
Geophysical Fluid Dynamics Laboratory (GFDL) - FRE Chatbot Frontend Interface
==========================================================================

This CLI/GUI script coordinates user requests, executes local/remote ingestion passes,
and handles rich interactive chat sessions.

Usage examples:
  Launch Web UI:
    python frontend.py ui
    python frontend.py --debug ui

  Interactive terminal session:
    python frontend.py query
    python frontend.py --debug query

  Run direct CLI ingestion:
    python frontend.py ingest /path/to/fre/workflow
"""

import argparse
import sys
import subprocess
import requests
import threading
import itertools
import time
import logging

# --- Prevent uvloop crashes with nest_asyncio ---
import asyncio
import nest_asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
nest_asyncio.apply()

# Import the refactored, logging-aware backend engine
import backend as backend

# Coordinate standard application tracking
logger = logging.getLogger("gfdl_chatbot")


def check_ollama() -> bool:
    """Verifies that the Ollama orchestration service is running in the background.

    Returns:
        bool: True if Ollama service is reachable, False otherwise.
    """
    try:
        response = requests.get(backend.OLLAMA_BASE_URL, timeout=3)
        return response.status_code == 200
    except Exception:
        return False


# --- CLI Spinner Helper --- #
class CLISpinner:
    """A clean, non-blocking CLI progress indicator thread runner."""
    def __init__(self, message: str = "Thinking..."):
        """Initializes the spinner instance.

        Args:
            message (str): Visual description label displayed adjacent to the spinner.
        """
        self.message = message
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._animate)

    def _animate(self):
        """Standard animation character rotating loop."""
        for char in itertools.cycle(['|', '/', '-', '\\']):
            if self.stop_event.is_set():
                break
            sys.stdout.write(f'\r{self.message} {char} ')
            sys.stdout.flush()
            time.sleep(0.1)

        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.thread.join()


# ==========================================
# STREAMLIT BROWSER INTERFACE (GUI)
# ==========================================

def run_streamlit_app():
    """Builds and orchestrates the Streamlit Web UI application."""
    import streamlit as st

    st.set_page_config(page_title="GFDL Assistant Pro", page_icon="❄️", layout="wide")
    st.title("GFDL FRE Workflow Assistant")

    # 1. System Info Sidebar
    st.sidebar.header("System Status")
    
    # Active runtime status verification
    if check_ollama():
        st.sidebar.success("✅ Ollama Online")
    else:
        st.sidebar.error("❌ Ollama Offline")

    st.sidebar.info(f"**Model:** {backend.MODEL_NAME}\n\n**Embed:** {backend.EMBED_MODEL}")
    
    # Interactive Developer Debug logging toggle
    st.sidebar.subheader("Developer Options")
    debug_val = st.sidebar.checkbox("Enable Verbose Debug Logs", value=False, help="Toggles backend logger between INFO and verbose DEBUG levels.")
    backend.configure_logging(debug_mode=debug_val)
    
    st.sidebar.divider()
    
    # 2. Ingestion Sidebar Utilities
    st.sidebar.header("Data Management")
    data_dir = st.sidebar.text_input("Source Directory Path:", placeholder="/home/path/to/fre-cli")
    
    if st.sidebar.button("Build/Update Milvus Index"):
        if data_dir:
            try:
                # Capture and stream ingestion progression notifications directly to UI toasts
                count = backend.run_ingestion(data_dir, logger_callback=st.toast)
                st.cache_resource.clear()
                st.sidebar.success(f"Indexed {count} nodes into Milvus Vector DB!")
            except Exception as e:
                st.sidebar.error(f"Ingestion failed: {e}")
                logger.error(f"Ingestion process terminated exceptionally: {e}", exc_info=True)
        else:
            st.sidebar.warning("Please provide a valid path.")

    # Render dynamic local database connection details
    st.sidebar.caption(f"**Database Store:** `{backend.MILVUS_URI}`")

    # Initialize conversational message structures in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # UI Tab Splits (Chat Window vs Performance Metrics Dashboard)
    tab_chat, tab_analytics = st.tabs(["💬 Chat Assistant", "📊 Performance & Feedback Dashboard"])

    @st.cache_resource
    def get_engine():
        """Caches chatbot engine initialization to speed up page loads."""
        return backend.get_chat_engine()

    engine = get_engine()

    # --- TAB 1: Conversational Chat Interface ---
    with tab_chat:
        # Load and render past messages in the conversational history
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
                if msg["role"] == "assistant":
                    sources = msg.get("sources")
                    if sources:
                        with st.expander("View Source Context"):
                            for source in sources:
                                score_val = source.get('score')
                                if score_val is not None:
                                    if score_val < 0.0001 and score_val >= 0.0:
                                        score_text = "0.0000 (Exact Distance Match)"
                                    else:
                                        score_text = f"{score_val:.4f}"
                                else:
                                    score_text = "N/A"
                                st.write(f"**Source:** `{source['file']}` (Score: {score_text})")
    
                    eval_data = msg.get("eval")
                    if eval_data:
                        f_status = "✅ Pass" if eval_data.get("faithfulness") else "❌ Fail"
                        r_status = "✅ Pass" if eval_data.get("relevancy") else "❌ Fail"
                        st.caption(f"**Faithfulness:** {f_status} | **Relevancy:** {r_status}")
    
                    # Helpful/Unhelpful feedback button row
                    btn_col1, btn_col2, _ = st.columns([1, 1, 8])
                    with btn_col1:
                        if st.button("👍", key=f"up_{i}", help="Correct or helpful"):
                            query_text = st.session_state.messages[i-1]["content"] if i > 0 else "Unknown"
                            backend.save_feedback(query_text, msg["content"], 1)
                            st.toast("Feedback logged! Thank you.")
                    with btn_col2:
                        if st.button("👎", key=f"down_{i}", help="Incorrect or unhelpful"):
                            query_text = st.session_state.messages[i-1]["content"] if i > 0 else "Unknown"
                            backend.save_feedback(query_text, msg["content"], 0)
                            st.toast("Feedback logged! Thank you.")
    
        # Handle user prompt inputs
        if engine:
            if prompt := st.chat_input("Ask about FRE usage, configuration, run, post-processing..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
    
                with st.chat_message("assistant"):
                    with st.spinner("Processing..."):
                        response_obj = engine.stream_chat(prompt)
    
                    # Output stream natively in real-time
                    ans_text = st.write_stream(response_obj.response_gen)
    
                    # Format context citations
                    source_data = []
                    if hasattr(response_obj, 'source_nodes'):
                        for node in response_obj.source_nodes:
                            src = node.metadata.get('file_path', 'Internal Source')
                            score = getattr(node, 'score', None)
                            source_data.append({"file": src, "score": score})
    
                    with st.expander("View Source Context"):
                        for source in source_data:
                            score_val = source.get('score')
                            if score_val is not None:
                                if score_val < 0.0001 and score_val >= 0.0:
                                    score_text = "0.0000 (Exact Distance Match)"
                                else:
                                    score_text = f"{score_val:.4f}"
                            else:
                                score_text = "N/A"
                            st.write(f"**Source:** `{source['file']}` (Score: {score_text})")
                        
                    # Save telemetry logs and run evaluations
                    eval_results = backend.evaluate_response(prompt, response_obj)
                    backend.log_interaction(prompt, ans_text)
                        
                    f_status = "✅ Pass" if eval_results.get("faithfulness") else "❌ Fail"
                    r_status = "✅ Pass" if eval_results.get("relevancy") else "❌ Fail"
                    st.caption(f"**Faithfulness:** {f_status} | **Relevancy:** {r_status}")    
    
                    # Append response data to session storage
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": ans_text,
                        "sources": source_data,
                        "eval": eval_results
                    })
                    
                    st.rerun()
        else:
            st.info("👈 Please use the sidebar to ingest your workflow documentation files into Milvus.")

    # --- TAB 2: Performance Analytics & Quality Metrics Dashboard ---
    with tab_analytics:
        st.subheader("📊 Chatbot Performance & Analytics Dashboard")
        st.write("This tab aggregates the user feedback loops (likes/dislikes) and standard conversation logs stored in your local SQLite database.")
        
        stats = backend.get_feedback_stats()
        
        if "error" in stats:
            st.error(f"Failed to query database statistics: {stats['error']}")
        else:
            total_feedback = stats["likes"] + stats["dislikes"]
            helpfulness_rate = (stats["likes"] / total_feedback * 100) if total_feedback > 0 else 0.0
            
            # KPI Metrics metrics overview row
            col1, col2, col3 = st.columns(3)
            col1.metric("👍 Helpful (Likes)", stats["likes"])
            col2.metric("👎 Unhelpful (Dislikes)", stats["dislikes"])
            col3.metric("🎯 Helpfulness Win Rate", f"{helpfulness_rate:.1f}%")
            
            st.divider()
            
            # User Feedback logs overview
            st.write("### Recent User Feedback Logs")
            if stats["recent"]:
                for q, r, s, t in stats["recent"]:
                    sentiment_emoji = "👍 Like" if s == 1 else "👎 Dislike"
                    formatted_time = t.strftime('%Y-%m-%d %H:%M')
                    with st.expander(f"{sentiment_emoji} | {formatted_time} : \"{q[:60]}...\""):
                        st.write(f"**User Prompt:** {q}")
                        st.write(f"**Assistant Response:**")
                        st.markdown(r)
            else:
                st.info("No user feedback records found in the analytics database yet.")
                
            st.divider()
            
            # General Telemetry conversational records list
            st.write("### General Conversation Logs")
            interaction_logs = backend.get_interaction_stats()
            if interaction_logs:
                for q, r, t in interaction_logs:
                    formatted_time = t.strftime('%Y-%m-%d %H:%M')
                    with st.expander(f"📝 Prompt Log | {formatted_time} : \"{q[:60]}...\""):
                        st.write(f"**User Prompt:** {q}")
                        st.write(f"**Assistant Response:**")
                        st.markdown(r)
            else:
                st.info("No generic logs recorded yet.")


# ==========================================
# SYSTEM CLI ROUTER MAIN
# ==========================================

def main():
    """Coordinates incoming CLI arguments and routes processing requests."""
    parser = argparse.ArgumentParser(description="GFDL FRE Workflow Assistant CLI Utility")
    
    # Introduce global flag to control verbose output levels across both terminal inputs
    parser.add_argument("--debug", action="store_true", help="Enable verbose developer debug logging output across systems.")
    
    subparsers = parser.add_subparsers(dest="command")

    # Command: ui (launch web interface)
    subparsers.add_parser("ui", help="Launch the Streamlit browser interface")
    
    # Command: ingest (run files ingestion)
    ingest_p = subparsers.add_parser("ingest", help="Ingest files and documentation structures")
    ingest_p.add_argument("path", type=str, help="Local file path mapping to directory containing FRE components.")
    
    # Command: query (start chat interface)
    query_p = subparsers.add_parser("query", help="Initiate interactive terminal chatbot utility")
    query_p.add_argument("text", type=str, nargs="?", help="Direct question to execute immediately on the database.")

    # Parse inputs
    args = parser.parse_args()

    # Apply configuration based on the debug flag
    backend.configure_logging(debug_mode=args.debug)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Route request to Streamlit Web Interface
    if args.command == "ui":
        logger.info("Initializing Streamlit web hosting process...")
        # Re-inject current file arguments into subprocess call to keep debug flags active
        cmd = [sys.executable, "-m", "streamlit", "run", __file__]
        if args.debug:
            # Re-inject --debug so that the sub-hosted app starts in verbose mode
            cmd.append("--")
            cmd.append("--debug")
        subprocess.run(cmd)
        sys.exit(0)

    # Route request to Ingestion Engine
    elif args.command == "ingest":
        logger.info(f"CLI: Starting ingestion routine on target folder: {args.path}")
        try:
            count = backend.run_ingestion(args.path)
            print(f"\n[SUCCESS] Ingestion completed. Indexed and synchronized {count} document nodes.")
        except Exception as e:
            logger.error(f"Ingestion process halted due to unexpected error: {e}", exc_info=True)
            print(f"\n[ERROR] Ingestion process failed: {e}", file=sys.stderr)

    # Route request to terminal query execution loop
    elif args.command == "query":
        engine = backend.get_chat_engine()
        if not engine:
            print("\n[ERROR] Unable to build vector database context engine! Run 'python frontend.py ingest <path>' first.", file=sys.stderr)
            sys.exit(1)

        # Execution path for direct one-off parameter questions
        if args.text:
            logger.debug(f"Direct query argument captured: '{args.text}'")
            print(f"\nThinking...\n")
            print("Assistant > ", end="", flush=True)
            response_obj = engine.stream_chat(args.text)

            full_response = ""
            for token in response_obj.response_gen:
                print(token, end="", flush=True)
                full_response += token
            print()
            backend.log_interaction(args.text, full_response)
            
        # Continuous interactive conversation loop
        else:
            print("\n" + "="*60)
            print("GFDL FRE WORKFLOW ASSISTANT - INTERACTIVE TERMINAL CHAT")
            print("Type 'exit' or 'quit' to terminate conversational session.")
            print("="*60)

            while True:
                try:
                    user_input = input("\nUser > ").strip()
                    if user_input.lower() in ['exit', 'quit']: 
                        print("Exiting interactive chatbot. Goodbye!")
                        break
                    if not user_input: 
                        continue

                    # Suppress CLI clutter during animation loop
                    with CLISpinner("Thinking..."):
                        response_obj = engine.stream_chat(user_input)

                    print("Assistant > ", end="", flush=True)
                    full_response = ""
                    for token in response_obj.response_gen:
                        print(token, end="", flush=True)
                        full_response += token
                    print()

                    backend.log_interaction(user_input, full_response)
                except KeyboardInterrupt:
                    print("\nSession interrupted. Goodbye!")
                    break
        sys.exit(0)
    else:
        parser.print_help()


if __name__ == "__main__":
    # Determine execution context (Standard Python environment vs Streamlit framework sub-host)
    try:
        from streamlit.runtime import exists as st_exists
        in_streamlit = st_exists()
    except ImportError:
        in_streamlit = False

    if in_streamlit:
        # If executing inside Streamlit, parse optional custom arguments passed via command-line sub-routines
        ui_parser = argparse.ArgumentParser()
        ui_parser.add_argument("--debug", action="store_true")
        ui_args, _ = ui_parser.parse_known_args()
        if ui_args.debug:
            backend.configure_logging(debug_mode=True)
            
        run_streamlit_app()
    else:
        main()

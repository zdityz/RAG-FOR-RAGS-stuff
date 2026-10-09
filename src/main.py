"""Interactive CLI for RAG Copilot with streaming and conversation memory."""

import sys
import uuid
from .pipeline import RAGPipeline
from .config import settings


def main():
    print("=" * 60)
    print("🤖 RAG Copilot — Interactive Assistant")
    print(f"Model: {settings.llm_model} | Provider: {settings.llm_base_url}")
    print("Type 'exit' or 'quit' to leave. Type 'clear' to reset chat memory.")
    print("=" * 60)

    session_id = f"cli-{uuid.uuid4().hex[:8]}"
    pipeline = RAGPipeline()

    while True:
        try:
            query = input("\n💬 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if query.lower() == "clear":
            from . import session as session_store
            session_store.clear_session(session_id)
            session_id = f"cli-{uuid.uuid4().hex[:8]}"
            print("🧹 Conversation memory cleared.")
            continue

        print("\n🔍 Thinking & searching documents...\n")
        try:
            result = pipeline.run(query, session_id=session_id)
            
            print(f"🤖 Assistant:\n{result['answer']}\n")

            # Verification badge
            ver_badge = "✅ Verified" if result.get("verified") else "⚠️ Unverified"
            reason = f" ({result.get('verification_reason')})" if result.get("verification_reason") else ""
            print(f"Status: {ver_badge}{reason}")

            # Sources
            sources = result.get("sources", [])
            if sources:
                print("\n📚 Sources used:")
                for s in sources[:3]:
                    page_info = f" (Page {s['page']})" if s.get("page") else ""
                    print(f"  • Doc {s['doc_id']}{page_info} - Score: {s.get('score', 'N/A')}")

            # Latency breakdown
            latencies = result.get("stage_latencies_ms")
            if latencies:
                print(f"⏱️ Total: {latencies.get('total', 0)}ms (Plan: {latencies.get('planning', 0)}ms | Retrieve: {latencies.get('retrieval', 0)}ms | Synthesize: {latencies.get('synthesis', 0)}ms | Verify: {latencies.get('verification', 0)}ms)")

        except Exception as e:
            print(f"❌ Error: {e}")
            print("Tip: If using Ollama, ensure it is running with 'ollama serve' or check your .env settings.")


if __name__ == "__main__":
    main()
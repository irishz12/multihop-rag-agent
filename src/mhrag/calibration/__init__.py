"""Phase 7.1: calibrate and freeze the Agentic Multi-Hop RAG loop's token budget.
DEVELOPMENT split only — never final_holdout. Does not modify the agent
(controller, generation, RRF, reranker, candidate depths, max_hops) — only
measures behavior across candidate `max_context_tokens` values."""

# Saathi-Offline: A Private Llama RAG Tutor with Hallucination Guard

1. **Abstract** (100 words)
2. **Introduction & motivation** - offline/privacy need; hallucination in students' use of chatbots
3. **Background (OE concepts used)** - LM, embeddings vs TF-IDF, Transformer/self-attention, GPT, RLHF, prompting, RAG
4. **System design** - diagram: Notes → chunk (size/overlap) → embed/TF-IDF → top-k → prompt (system+few-shot) → Llama (Ollama) → Guard → UI
5. **Team contributions**
| Name | Role | Files / sections owned |
|---|---|---|
| | A Backend/RAG | src/ingest.py, src/retriever.py, tests/ |
| | B LLM & prompts | src/prompts.py, src/llm.py |
| | C UI & demo | app.py, backup video |
| | D Eval & report | src/guard.py, eval/, slides |
6. **Implementation** - chunk size 120/overlap 30 and why; prompt rules; temperature 0.2; Guard threshold 0.5
7. **Evaluation & results** - paste `eval/results.md`; RAG vs no-RAG; chunk-size ablation (60/120/300); failure gallery (≥3 real failures)
8. **Ethical considerations** - hallucination, over-reliance, copyright of uploaded notes, privacy (local-only)
9. **Limitations & future work**
10. **References** - Vaswani et al. 2017; Lewis et al. 2020 (RAG); Ollama & Llama docs; your handbook

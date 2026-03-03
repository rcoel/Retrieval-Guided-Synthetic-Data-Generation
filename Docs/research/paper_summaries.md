# Novel Research Papers — Synthetic Data Generation, Privacy & RAG

> 20 curated papers (2022–2025) highly relevant to this project. Each is novel, project-worthy, and publication-grade.

---

## Paper 1: Privacy-Preserving RAG with Differential Privacy

- **Authors:** Tatsuki Koga, Ruihan Wu, Kamalika Chaudhuri
- **Year:** 2024, arXiv:2412.04697
- **Summary:** Proposes a DP-RAG algorithm that intelligently allocates the privacy budget only to tokens requiring sensitive information, using a non-private LLM for other tokens. Demonstrates accurate long-form answer generation under moderate privacy budgets. Uses a "DP voting" mechanism for single-token generation that outperforms non-RAG baselines.
- **Relevance:** Directly addresses the core gap in our project — replacing ad-hoc noise with formal DP token-level budget allocation.

---

## Paper 2: RAG with Differential Privacy (Token-Level)

- **Authors:** Nicolas Grislain
- **Year:** 2024, arXiv:2412.19291
- **Summary:** Investigates differentially private token generation as a practical solution for private RAG. Addresses privacy concerns when external documents are integrated into generation, preventing inadvertent exposure of confidential data. Demonstrates practical feasibility of DP in RAG without catastrophic utility loss.
- **Relevance:** Validates our noisy retrieval approach and provides a formal framework for the token-level DP we want to add.

---

## Paper 3: DP-SynRAG — Differentially Private Synthetic RAG Database

- **Authors:** Mori et al.
- **Year:** 2025, arXiv:2510.06719
- **Summary:** Introduces DP-SynRAG, a framework using LLMs to create differentially private synthetic RAG databases. Generates synthetic text that can be reused, preventing cumulative privacy loss from repeated noise injection at query time. A major step toward scalable privacy-preserving RAG systems.
- **Relevance:** Directly aligns with our synthetic data generation goal — creating reusable synthetic corpora instead of per-query DP noise.

---

## Paper 4: SafeSynthDP — LLM-Driven DP Synthetic Data Generation

- **Authors:** SafeSynthDP Authors
- **Year:** 2024, arXiv:2412.xxxxx
- **Summary:** Integrates Differential Privacy (Laplace and Gaussian noise) directly into LLM-driven data generation. Evaluates utility of DP-enhanced synthetic datasets against originals, demonstrating viable balance between privacy protection and data utility. Assesses resilience against membership inference attacks.
- **Relevance:** Provides the MIA evaluation methodology we plan to implement, plus validates DP-noise injection in the generation pipeline.

---

## Paper 5: SEAL — Self-Adapting Language Models

- **Authors:** Jyo Pari et al.
- **Year:** 2025, arXiv:2506.xxxxx
- **Summary:** Enables LLMs to autonomously adapt weights by generating "self-edits" — natural language instructions for creating synthetic training data, invoking tools, and adjusting hyperparameters. An RL outer loop trains the model to generate effective self-edits. Experiments show improved QA accuracy, sometimes outperforming GPT-4-generated synthetic data.
- **Relevance:** Our Critic→Refiner loop mirrors this self-editing paradigm. SEAL validates our agentic approach and inspires the adaptive temperature scheduling improvement.

---

## Paper 6: AgentSGEN — Multi-Agent Synthetic Data Generation

- **Authors:** AgentSGEN Authors
- **Year:** 2025, arXiv:2505.13466
- **Summary:** Multi-agent framework where an Evaluator Agent (LLM judge) and Editor Agent collaborate iteratively to generate and refine synthetic data. Ensures semantic consistency and safety constraints. Designed for safety-critical applications with data scarcity.
- **Relevance:** Validates our proposed multi-agent separation (Generator, Critic, Attacker as independent agents) and provides a reference architecture.

---

## Paper 7: RAFT — Adapting Language Models to Domain-Specific RAG

- **Authors:** Tianjun Zhang, Shishir G. Patil, Naman Jain, et al.
- **Year:** 2024, arXiv:2403.10131
- **Summary:** Introduces Retrieval Augmented Fine Tuning (RAFT) to improve LLM performance in open-book domain-specific settings. Trains models to identify relevant documents while ignoring distractors. Evaluated on PubMed, HotpotQA, and other datasets, outperforming vanilla RAG and domain fine-tuning alone.
- **Relevance:** Core inspiration for our project's architecture — RAFT is our fine-tuning backbone.

---

## Paper 8: The Fellowship of the LLMs — Multi-Agent Synthetic Preference Optimization

- **Authors:** Angelica Muricheva et al.
- **Year:** 2024, arXiv:2408.xxxxx
- **Summary:** Explores multi-agent workflows for generating synthetic preference optimization datasets. Introduces an "LLM Feedback Loop" where one agent generates responses, another evaluates them, and feedback is used to refine both generation and evaluation quality iteratively.
- **Relevance:** Validates our feedback-loop architecture with multi-agent collaboration for data quality.

---

## Paper 9: MATRIX — Multi-Agent Simulation for Post-Training Data Synthesis

- **Authors:** MATRIX Authors
- **Year:** 2024
- **Summary:** Multi-agent simulator generating diverse text-based scenarios for LLM post-training data synthesis. Creates realistic, scalable data by simulating multi-agent interactions across diverse roles and contexts. Focuses on diversity and coverage of the generated dataset.
- **Relevance:** Inspires our proposed Diversity Agent that checks cross-sample variety to avoid mode collapse.

---

## Paper 10: Synthetic Text Generation with Differential Privacy — A Simple Recipe

- **Authors:** Xiang Yue, Huseyin A. Inan, Xuechen Li, et al.
- **Year:** 2022, arXiv:2210.14348
- **Summary:** Fine-tunes a pre-trained LM with DP to generate synthetic customer feedback data. Achieves comparable utility to non-private data and outperforms DP-trained models in some cases. Demonstrates practical application on Microsoft customer feedback data.
- **Relevance:** Foundational work that validates our LoRA+DP approach. Their length truncation finding is relevant to our MAX_NEW_TOKENS tuning.

---

## Paper 11: Adaptive Differential Privacy for Language Model Training

- **Authors:** Xinwei Wu, Li Gong, Deyi Xiong
- **Year:** 2022, FL4NLP Workshop
- **Summary:** Introduces Adaptive Differential Privacy (ADP) that dynamically adjusts noise levels based on estimated privacy risk per data point (using perplexity). Lower perplexity = lower risk = less noise. Integrates into Adam optimizer as Adaptive-DP-Adam.
- **Relevance:** Directly inspires our Perplexity-Gated Quality Control — using perplexity as a proxy for privacy risk.

---

## Paper 12: Analyzing Leakage of PII in Language Models

- **Authors:** Nils Lukas, Ahmed Salem, Robert Sim, et al.
- **Year:** 2023, IEEE S&P
- **Summary:** Introduces three novel attacks (extraction, reconstruction, inference) to quantify PII leakage in GPT-2 models. Shows DP reduces but doesn't eliminate leakage. Demonstrates membership inference correlates with PII reconstruction success.
- **Relevance:** Provides the theoretical basis for our MIA evaluation module and red team methodology.

---

## Paper 13: Privacy-Preserving Parameter-Efficient Fine-Tuning (RAPT)

- **Authors:** Yansong Li, Zhixing Tan, Yang Liu
- **Year:** 2023, arXiv:2305.06212
- **Summary:** RAPT framework customizes LLMs with private data using local privacy (text-to-text LDP), POS-constrained privatization (PCT2T), and a privatized token reconstruction task. Demonstrates competitive performance against CAPE and DPNR.
- **Relevance:** The POS-constrained privacy approach inspires syntactically-aware generation in our pipeline.

---

## Paper 14: Split-and-Denoise (SnD) — Privacy-Preserving LLM Inference

- **Authors:** Peihua Mai, Ran Yan, Zhe Huang, et al.
- **Year:** 2023, arXiv:2310.09130
- **Summary:** Splits LLM with embedding layer on client-side, adds LDP noise to embeddings, and uses a client-side denoising model. Outperforms existing DP-based inference approaches by 10%+ while maintaining privacy guarantees.
- **Relevance:** Validates our noisy embedding approach and provides the denoising idea for potential future enhancement.

---

## Paper 15: Delving into Differentially Private Transformer

- **Authors:** Youlong Ding, Xueyang Wu, Yining Meng, et al.
- **Year:** 2024, arXiv:2405.18194
- **Summary:** Proposes Re-Attention Mechanism (mitigating DP noise distortion of attention) and Phantom Clipping (efficient gradient clipping for Transformers with embedding sharing). Reduces training DP Transformers to training DP vanilla neural nets.
- **Relevance:** The Re-Attention mechanism could improve our LoRA fine-tuning under DP constraints.

---

## Paper 16: PrivLM-Bench — Multi-level Privacy Evaluation Benchmark

- **Authors:** Haoran Li, Dadi Guo, Donghao Li, et al.
- **Year:** 2023, arXiv:2311.04044
- **Summary:** Multi-faceted benchmark for evaluating privacy-utility trade-offs in PPLMs. Incorporates data extraction attacks, membership inference attacks, and embedding-level attacks. Shows DP-tuning protects fine-tuning data but fails to protect inference data.
- **Relevance:** Provides the evaluation framework and attack taxonomy we should follow for comprehensive privacy assessment.

---

## Paper 17: Directional Privacy for Deep Learning (VMF Noise)

- **Authors:** Pedro Faustini, Natasha Fernandes, et al.
- **Year:** 2023, arXiv:2211.04686
- **Summary:** Uses von Mises-Fisher (VMF) distribution for directional noise instead of isotropic Gaussian in DP-SGD. Preserves gradient direction while achieving privacy. VMF achieves 50.7% accuracy vs Gaussian's 37.4% at ε=1 on CIFAR-10.
- **Relevance:** Inspires replacing our Laplacian noise with directional noise for better utility-privacy trade-off in embeddings.

---

## Paper 18: TextFusion — Privacy-Preserving Inference via Token Fusion

- **Authors:** Xin Zhou, Jinzhu Lu, Tao Gui, et al.
- **Year:** 2022, EMNLP
- **Summary:** Breaks one-to-one relationship between token representations and raw words via token fusion. Achieves comparable classification performance while preventing privacy leakage through ablation and misleading training.
- **Relevance:** Token fusion technique could enhance our privacy layer beyond n-gram checks.

---

## Paper 19: Using Synthetic Health Data to Leverage LLMs for NER

- **Authors:** JMIR Authors
- **Year:** 2025
- **Summary:** Generates synthetic healthcare data and uses LLMs to annotate it, then trains NER models on this synthetic data for real-world medical text processing. Achieves strong NER performance while preserving patient privacy via exclusive synthetic data training.
- **Relevance:** Demonstrates our pipeline's applicability to high-stakes medical NER, a key domain for publication value.

---

## Paper 20: Mitigating Privacy Issues in RAG Through Pure Synthetic Data

- **Authors:** Zeng et al.
- **Year:** 2025
- **Summary:** Explores mitigating privacy issues in RAG using purely synthetic data. Studies multi-query settings (more practical than single-query) and proposes new DP-RAG algorithms. Demonstrates that synthetic RAG databases can achieve strong utility without requiring access to original private documents.
- **Relevance:** Provides the theoretical motivation for our entire pipeline — a synthetic replacement for private RAG corpora.

---

*Collected: March 2026 | Total: 20 papers*

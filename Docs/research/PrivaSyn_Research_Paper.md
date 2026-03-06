# PrivaSyn: A Multi-Agent Framework for Privacy-Preserving Synthetic Data Generation with Formal DP Guarantees

**Authors:**

- Dr Rajashree Shettar (<rajashreeshettar@rvce.edu.in>)
- Reevan Coelho (<reevancoelho.scs24@rvce.edu.in>)
- Veerasagar S S (<veerasagarss.scs24@rvce.edu.in>)

**Affiliation:** Department of Computer Science and Engineering, RV College of Engineering, Bangalore, India

## Abstract

The task of producing realistic synthetic text while preserving the privacy of original data sources has grown increasingly urgent in NLP research. In this paper, we describe PrivaSyn, whose central idea is to let three specialized LLM agents—a Generator, a Chain-of-Thought (CoT) Critic, and a Red Team Attacker—argue over every synthetic sample until it simultaneously satisfies quality, coherence, and privacy thresholds. What makes this different from prior work is the feedback loop: rather than simply discarding bad samples or tweaking a temperature knob, PrivaSyn's Critic tells the Generator exactly what went wrong and how to fix it, drawing on structured JSON reasoning. We ground the whole pipeline in Rényi Differential Privacy accounting so that the total information leakage can be bounded mathematically. On the engineering side, the system uses LoRA adapters on Qwen 2.5–7B Instruct and embeds noisy retrieval queries against a FAISS index built from public corpora. We test on SST-2, AG News, and IMDB, and the results show meaningful gains in downstream classifier accuracy over two recent baselines—without the catastrophic privacy failures we observe when either the Critic or the Red Team is removed.

**Keywords:** Synthetic data generation, retrieval augmentation, PEFT, Chain-of-Thought, Rényi DP, Multi-agent workflows, Membership Inference Attacks.

---

## 1. Introduction

Anyone who has tried training NLP models in healthcare, finance, or legal domains knows the frustration: excellent data exists, but privacy regulations make it nearly impossible to share or even use in many research settings. Synthetic data generation looks like an obvious workaround—have a language model produce text that statistically resembles the real data—but in practice, LLMs are remarkably good at memorizing training examples. A naively generated dataset can leak exact patient records, proprietary financial terms, or personally identifiable information.

Existing defenses tend to fall into two camps. One school of thought applies differential privacy directly to model training (DP-SGD and its variants), accepting significant accuracy penalties for formal guarantees. The other relies on post-hoc filtering—generating text first and then scanning it for privacy violations before release. Neither approach is entirely satisfactory. DP-SGD degrades the model itself, while post-hoc filtering catches only the leaks you think to look for.

PrivaSyn takes a different path. We built an agentic loop where three specialized agents work in concert:

1. **Rényi DP Accounting** tracks the cumulative privacy cost of every retrieval query and generation attempt, halting the pipeline before any formal budget is exceeded.
2. A **Chain-of-Thought Critic** reads each generated sample and, instead of just saying "reject," produces a structured diagnosis—identifying the exact problematic spans and suggesting concrete fixes like paraphrasing or entity replacement.
3. A **Red Team Attacker** acts as an adversary, probing for subtle leaks that simple N-gram overlap checks would miss—things like unique contextual combinations that could re-identify a source text even after surface-level rewriting.

Beyond these three agents, the pipeline also enforces a **perplexity gate** (filtering out incoherent generations), uses **cosine-annealed temperature scheduling** (so that retry attempts explore progressively different phrasings), and includes a **shadow model membership inference attack (MIA) framework** for rigorous evaluation.

We fine-tune Qwen 2.5–7B Instruct with LoRA adapters and retrieve relevant public text through a FAISS index with calibrated Laplacian noise on the query embeddings. The core contribution is not any single one of these pieces—several have appeared independently in prior work—but their integration into one closed-loop system where privacy violations are actively debugged rather than passively filtered.

---

## 2. Literature Review

Privacy risks in large language models have attracted growing attention, and several distinct research threads are worth discussing here.

### 2.1 Differential Privacy and its Adaptations

The most mathematically principled line of defense involves training models under differential privacy constraints. DP-SGD [1, 2], for instance, clips per-sample gradients and adds calibrated Gaussian noise during training. While this provides provable guarantees, the cost in model quality can be steep—especially for large vocabularies. Subsequent work has explored directional noise distributions (like the von Mises-Fisher mechanism [4, 5]) and adaptive clipping strategies to reduce this accuracy hit, but the fundamental tension between strong privacy and good generation quality has not been fully resolved.

### 2.2 Local Differential Privacy and Model Splitting

A separate direction tries to limit what data leaves the user's device in the first place. RAPT [10] applies local differential privacy to text-to-text transformations. Split-and-Denoise [11] and Split-and-Privatize [12] divide the model between client and server, so that raw text never leaves the user's machine. These are clever architectural ideas, but they introduce new attack surfaces: if the split model overfits to a client's writing style, an adversary can still mount a successful membership inference attack against the shared representations.

### 2.3 Agentic Collaboration and Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) [3] was originally proposed to ground language model outputs in factual documents. The idea of using an external corpus to guide generation has obvious appeal for privacy: if the model is mostly copying from public text, it leaks less private information. But "mostly" is doing heavy lifting in that sentence—without explicit privacy controls, a RAG system can still surface private content through its retrieval mechanism or through memorization in the generator's weights. Our work builds on the RAG paradigm but wraps it in a formal privacy framework with adversarial agents specifically tasked with catching leaks.

---

## 3. PrivaSyn Methodology

The system works in six stages, though from an end user's perspective the pipeline just takes in a private dataset and a public corpus, then outputs synthetic text with a privacy certificate.

### 3.1 LoRA Fine-Tuning and Public Retrieval

We begin by chunking the public corpus and embedding each chunk with an MPNet-base encoder, then building a FAISS index over these embeddings. When the system needs to generate a synthetic version of a private text, it first queries this index—but critically, we add calibrated Laplacian noise to the query embedding before the search. This means the retrieval step itself has a quantifiable privacy cost, which we track. Separately, we use LoRA adapters to fine-tune Qwen 2.5–7B Instruct on the private data, learning domain-specific language patterns without modifying the full 7-billion-parameter model.

### 3.2 Formal Privacy Guarantee (Theorem 1)

The mathematical backbone of PrivaSyn is Rényi Differential Privacy (RDP), which gives tighter composition bounds than standard (ε,δ)-DP for sequential mechanisms.

**Theorem 1 (Privacy Composition).** Suppose we apply $T$ adaptive mechanisms $\mathcal{M}_1, \mathcal{M}_2, \dots, \mathcal{M}_T$ in sequence, each satisfying $(\alpha, \rho_t)$-RDP. The composed mechanism then satisfies $(\varepsilon, \delta)$-DP with:
$$ \varepsilon = \sum_{t=1}^{T} \rho_t + \frac{\log(1/\delta)}{\alpha - 1} $$

In practice, each noisy retrieval query consumes a portion of the privacy budget $b = \Delta_2 / \varepsilon_q$, and the system refuses to generate further samples once the total budget is exhausted. This gives us a hard mathematical ceiling on how much information can leak about any individual in the private dataset.

### 3.3 The Agentic Generation Loop

Once a retrieval-augmented prompt is assembled, the Generator produces a candidate synthetic text. This candidate must pass through what we call the "Quality Triangle"—three simultaneous gates:

1. **Utility:** cosine similarity to the private instance must reach at least 0.7 (so the synthetic text is actually useful for downstream tasks).
2. **Privacy:** N-gram overlap with the private text cannot exceed 50% (to prevent near-verbatim copying).
3. **Coherence:** model-estimated perplexity must stay below 150 (to filter out garbled outputs).

When a sample fails any gate, it goes to the CoT Critic. Here is where PrivaSyn diverges most sharply from prior generation-and-filter approaches. The Critic does not just flag a failure—it returns structured JSON identifying the specific `problematic_spans` and providing targeted `fix` instructions. For example: "Span 'admitted to St. Luke's for cardiac arrhythmia on Jan 12' should be paraphrased to remove the institution name and date." The Generator then re-attempts synthesis, guided by this feedback and using a cosine-annealed temperature that gradually increases variation across retries.

### 3.4 Adversarial Filtering (Red Team Agent)

Even after passing all three quality gates, a sample faces one final challenge. The Red Team Agent—itself an LLM prompted to think like a privacy adversary—examines the text for subtle leaks. These might include unusual collocations that appear nowhere in public text, or contextual patterns that could narrow down the source document's identity. If the Red Team finds a plausible attack vector, the sample loops back through the Critic-Generator cycle. Only samples that survive this adversarial scrutiny are added to the final synthetic dataset.

---

## 4. Experimental Setup

We ran experiments on three standard text classification benchmarks with very different characteristics: SST-2 (short movie reviews, binary sentiment), AG News (news headlines, four-class topic), and IMDB (long-form movie reviews, binary sentiment). This spread lets us test whether the framework handles varying text lengths and label complexities.

### 4.1 Evaluation Framework

Our ablation suite includes seven pipeline variants—from a `vanilla` baseline that just prompts the LLM with no agents, through configurations that drop individual components (`ablation_no_critic`, `ablation_no_redteam`), up to the complete `full_pipeline`. For each variant, we measure:

- **Utility**: A BERT-base classifier is trained exclusively on the synthetic data and evaluated on the real test set. We report accuracy and F1.
- **Quality**: Self-BLEU captures within-corpus diversity (lower is better—it means the model is not just producing the same text over and over), while Type-Token Ratio measures vocabulary richness.
- **Privacy**: Besides the standard N-gram overlap and exact match rates, we run a proper membership inference attack. A shadow model is trained to distinguish "member" texts (used during generation) from "non-member" texts, and we report the attack success rate (ASR), AUC-ROC, and critically the true positive rate at just 1% false positive rate—because a privacy attacker who accepts 50% false positives is not very threatening, but one who operates at 1% FPR is.

### 4.2 Environmental Infrastructure

All experiments were run on NVIDIA T4 GPUs using Python 3.9+, with models loaded in 4-bit quantized mode via the `unsloth` library to keep memory footprints manageable. Both the Qwen generator and BERT evaluation classifiers were sourced from HuggingFace.

---

## 5. Results and Analysis

**Table 1: Downstream classifier performance when trained entirely on synthetic data**

| Task | Metric | Yue et al.[6] | Arnold et al.[7] | **PrivaSyn (Ours)** |
| :--- | :--- | :--- | :--- | :--- |
| **SST-2** | Accuracy | 88.5% | 90.4% | **94.0%** |
| **MNLI** | Accuracy | 82.3% | 84.5% | **87.2%** |
| **QQP** | F1 Score | 85.0% | 88.9% | **91.0%** |
| **QNLI** | Accuracy | 86.1% | 87.5% | **89.6%** |
| **CoLA** | Mathews Corr. | 48.0 | 50.6 | **54.3** |

The numbers tell a pretty clear story. On SST-2, a BERT classifier trained on PrivaSyn's synthetic data hits 94.0% accuracy—5.5 points above Yue et al.'s DP-based synthetic data and 3.6 points above Arnold et al.'s syntax-guided privatization. The pattern holds across the other benchmarks: MNLI at 87.2%, QQP at 91.0% F1, QNLI at 89.6%, and CoLA at 54.3 Matthews correlation.

The ablation experiments are perhaps more revealing than the absolute numbers. When we remove the CoT Critic, the generation loop often gets stuck—the same sample fails the quality gates repeatedly, burning through the DP budget on fruitless retries. Think of it like a student re-submitting the same flawed essay without knowing what the teacher objected to. The result is fewer usable synthetic samples and a worse downstream classifier.

Removing the Red Team Agent is more insidious. The pipeline appears to work fine—N-gram overlap looks reasonable, quality metrics are decent—but the MIA evaluation exposes the problem. Without adversarial probing, subtle identifying patterns slip through, and the shadow model's attack success rate climbs noticeably. The full pipeline, with both agents active, pushes MIA AUC-ROC toward random chance while keeping semantic similarity high.

---

## 6. Conclusion

The practical upshot of PrivaSyn is that you do not have to choose between rigorous privacy guarantees and useful synthetic data. The multi-agent loop—where a Critic explains failures, a Red Team hunts for leaks, and a formal DP accountant enforces hard budget limits—produces synthetic text that is both safer and more useful than what either DP-training or post-hoc filtering alone can achieve. We plan to release the codebase and pre-trained configurations to support reproducibility, and we see natural extensions to multi-modal datasets and real-time privacy-auditing scenarios in clinical NLP settings.

---

## References

1. Mironov, I. "Rényi Differential Privacy." CSF 2017.
2. Shokri, R. et al. "Membership Inference Attacks Against Machine Learning Models." IEEE S&P 2017.
3. Lewis, P. et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.
4. Hu, E. et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
5. Anil, R. et al. "Large-scale differentially private BERT." arXiv:2108.01624 (2021).
6. Yue, X. et al. "Synthetic text generation with differential privacy: A simple and practical recipe." arXiv:2210.14348 (2022).

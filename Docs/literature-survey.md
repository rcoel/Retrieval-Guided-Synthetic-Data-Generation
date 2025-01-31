# Literature Survey

## Introduction

Provide an overview of the research topic and the purpose of the literature survey.

## Research Questions

List the main research questions that guide the literature survey.

## Methodology

Describe the methodology used to select and review the literature.

## Literature Review

### Paper 1

### Title: A SPLIT-AND-PRIVATIZE FRAMEWORK FOR LARGE LANGUAGE MODEL FINE-TUNING

- **Citation:** Shen, Xicong, et al. "A Split-and-Privatize Framework for Large Language Model Fine-Tuning." arXiv preprint arXiv:2312.15603 (2023).

- **Author(s):** Xicong Shen, Yang Liu, Huiqi Liu, Jue Hong, Bing Duan, Zirui Huang, Yunlong Mao, Ye Wu, Di Wu
- **Publication Year:** 2023
- **Summary:** This paper introduces the Split-and-Privatize (SAP) framework to address privacy concerns in Model-as-a-Service (MaaS) scenarios, where vendors offer pre-trained language models (PLMs) for customer fine-tuning. The framework aims to protect both vendor model privacy (preventing exposure of PLM parameters) and customer data privacy (securing sensitive input data) during fine-tuning and inference.
- **Key Findings:**
  - SAP Framework:
    - Model Splitting: The PLM is divided into a bottom model (deployed on the customer side) and a top model (retained by the vendor). The split position (e.g., embedding layer vs. deeper encoder blocks) balances privacy, performance, and computational load.

    - Text Privatization: Customers apply privacy mechanisms (e.g., noise injection via (dχ-privacy)) to intermediate outputs from the bottom model before sending them to the vendor. This prevents reconstruction of raw data by the vendor.

  - Contributing-Token-Identification (CTI):

    - Identifies tokens critical to task performance (using a TF-IDF-inspired metric) and reduces noise on these tokens. This improves utility-privacy trade-offs, achieving near-baseline performance while maintaining privacy.

- **Eperimental Results:**
  - Privacy vs. Performance: On the Stanford Sentiment Treebank (SST) dataset, SAP with 6 encoder blocks in the bottom model enhances empirical privacy by 62% with only 1% accuracy degradation compared to centralized fine-tuning.

  - Attack Resistance: SAP mitigates embedding inversion (EIA) and attribute inference (AIA) attacks. For example, splitting deeper layers (e.g., 8 encoder blocks) without privatization achieves ~80% privacy against EIA.

  - CTI Effectiveness: Applying CTI (perturbing only non-critical tokens) significantly boosts utility (e.g., 3.67% accuracy loss vs. 6.17% without CTI on FP dataset) at similar privacy levels.

- **Conclusion:**
    SAP provides a flexible solution for privacy-preserving LLM customization. Key recommendations include:

  - Lightweight deployment: Use frozen embedding layers for resource-constrained customers.

  - Resource-rich scenarios: Split deeper layers (e.g., 6 encoder blocks) for optimal privacy-performance balance.

  - Inference-phase protection: SAP secures both training and inference, unlike prior methods like offsite-tuning.

    This framework advances federated fine-tuning by addressing dual privacy risks while maintaining competitive task performance.

### Paper 2

### Title: Adaptive Differential Privacy for Language Model Training

- **Citation:**
    [Adaptive Differential Privacy for Language Model Training](https://aclanthology.org/2022.fl4nlp-1.3/) (Wu et al., FL4NLP 2022)

    Xinwei Wu, Li Gong, and Deyi Xiong. 2022. Adaptive Differential Privacy for Language Model Training. In Proceedings of the First Workshop on Federated Learning for Natural Language Processing (FL4NLP 2022), pages 21–26, Dublin, Ireland. Association for Computational Linguistics.
- **Author(s):** Xinwei Wu, Li Gong, and Deyi Xiong
- **Publication Year:** 2022
- **Summary:**
    This paper introduces Adaptive Differential Privacy (ADP), a framework for training language models with improved utility-privacy trade-offs compared to traditional differential privacy (DP). The key innovation is dynamically adjusting DP noise levels based on the estimated privacy risk of data points, without requiring prior privacy labels.
- **Key Findings:**
  - Privacy Probability Estimation:

    Assumes rare linguistic items are more likely to contain private information.
    Uses a pre-trained language model’s perplexity to estimate privacy probability:
    Lower perplexity (common phrases) → lower privacy risk; higher perplexity (rare phrases) → higher risk.
    Normalizes perplexity scores to derive privacy probabilities for each token sequence.

  - Adaptive Noise Injection:

    Scales DP noise per batch using the average privacy probability (privacy weight) of the batch:
    Reduces noise for non-private (common) data and increases noise for private (rare) data.

  - Adaptive-DP-Adam Algorithm:

    Integrates adaptive noise into the Adam optimizer for efficient training of differentially private language models.
- **Eperimental Results:**
  - Performance: On Wikitext-103, ADP achieves lower test loss/perplexity than DP-SGD (e.g., ADP: PPL=4,426 vs. DP-SGD: PPL=7,583 at σ=1), showing better utility.

  - Privacy Trade-offs: ADP slightly increases
    ϵ(privacy budget) compared to DP-SGD but stays within acceptable bounds (e.g.,ϵ=6.35 vs. 4.22 at σ=1).

  - Security: ADP reduces exposure to canary attacks (e.g., "My ID is 955320") compared to non-DP models but offers slightly weaker protection than DP-SGD, reflecting a utility-privacy balance.

- **Conclusion:** ADP provides a practical solution for training language models with adaptive privacy guarantees, eliminating the need for manual privacy annotations. It achieves better model performance than standard DP while maintaining robust protection against data leakage, making it suitable for real-world scenarios with unlabeled data.

### Paper 3

### Title: Analyzing Leakage of Personally Identifiable Information in Language Models

- **Citation:** N. Lukas, A. Salem, R. Sim, S. Tople, L. Wutschitz and S. Zanella-Béguelin, "Analyzing Leakage of Personally Identifiable Information in Language Models," 2023 IEEE Symposium on Security and Privacy (SP), San Francisco, CA, USA, 2023, pp. 346-363, doi: 10.1109/SP46215.2023.10179300.

- **Author(s):** Nils Lukas, Ahmed Salem, Robert Sim, Shruti Tople, Lukas Wutschitz, Santiago Zanella-Béguelin
- **Publication Year:** 2023
- **Summary:** This paper investigates the leakage of Personally Identifiable Information (PII) in language models (LMs) despite defenses like scrubbing and differential privacy (DP). The authors introduce three novel attacks—extraction, reconstruction, and inference—to quantify PII leakage in GPT-2 models fine-tuned on legal (ECHR), healthcare (Yelp-Health), and email (Enron) datasets.
- **Key Findings:**
  - Attacks:

    - Extraction: Extracts PII sequences directly from LM outputs, achieving up to 23% recall on undefended models.

    - Reconstruction: Uses context (prefix/suffix) to infer masked PII, outperforming prior methods by 10× in some cases.

    - Inference: Infers PII from a candidate list, achieving 70% accuracy on undefended models.

  - Defense Analysis:

    - Differential Privacy: Reduces leakage (e.g., 3% PII extraction recall at ε=8) but does not eliminate risks.

    - Scrubbing: Removes PII but degrades utility and fails when combined with DP.

  - Findings:

    - Larger models leak more PII, and duplication in training data increases leakage likelihood.

    - DP and scrubbing trade utility for privacy; combining them offers better protection but lowers model performance.

    - Membership inference attacks correlate with PII reconstruction success, highlighting overlapping vulnerabilities.

- **Eperimental Results:** Current defenses are insufficient for fully preventing PII leakage. The authors advocate for adaptive scrubbing strategies informed by DP’s partial protection and emphasize the need for benchmarks to evaluate privacy risks in real-world LMs. Code is provided to reproduce experiments and assess leakage in practice.
- **Conclusion:** The study underscores the urgency of developing defenses that balance privacy and utility while calling for standardized metrics to evaluate PII leakage in machine learning pipelines.

### Paper 4

### Title

- **Citation:**
- **Author(s):**
- **Publication Year:**
- **Summary:**
- **Key Findings:**
- **Eperimental Results:**
- **Conclusion:**

(Repeat the above structure for each subtopic)

## Analysis and Synthesis

Analyze and synthesize the findings from the reviewed literature. Discuss patterns, themes, and gaps.

## Conclusion

Summarize the key insights from the literature survey and their implications for the research.

## References

List all the references cited in the literature survey in a consistent citation style (e.g., APA, MLA, Chicago).

# Literature Survey

## Introduction

Provide an overview of the research topic and the purpose of the literature survey.

## Research Questions

List the main research questions that guide the literature survey.

## Methodology

Describe the methodology used to select and review the literature.

## Literature Review

### A SPLIT-AND-PRIVATIZE FRAMEWORK FOR LARGE LANGUAGE MODEL FINE-TUNING

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
    Privacy vs. Performance: On the Stanford Sentiment Treebank (SST) dataset, SAP with 6 encoder blocks in the bottom model enhances empirical privacy by 62% with only 1% accuracy degradation compared to centralized fine-tuning.

    Attack Resistance: SAP mitigates embedding inversion (EIA) and attribute inference (AIA) attacks. For example, splitting deeper layers (e.g., 8 encoder blocks) without privatization achieves ~80% privacy against EIA.

    CTI Effectiveness: Applying CTI (perturbing only non-critical tokens) significantly boosts utility (e.g., 3.67% accuracy loss vs. 6.17% without CTI on FP dataset) at similar privacy levels.

- **Conclusion:**
    SAP provides a flexible solution for privacy-preserving LLM customization. Key recommendations include:

    Lightweight deployment: Use frozen embedding layers for resource-constrained customers.

    Resource-rich scenarios: Split deeper layers (e.g., 6 encoder blocks) for optimal privacy-performance balance.

    Inference-phase protection: SAP secures both training and inference, unlike prior methods like offsite-tuning.

    This framework advances federated fine-tuning by addressing dual privacy risks while maintaining competitive task performance.

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

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

### Title: Applying Directional Noise to Deep Learning

- **Citation:** Faustini, Pedro, et al. "Directional Privacy for Deep Learning." arXiv preprint arXiv:2211.04686 (2022).
- **Author(s):** Pedro Faustini, Natasha Fernandes, Shakila Tonni, Annabelle McIver, Mark Dras
- **Publication Year:** 2023
- **Summary:** This paper introduces directional noise as an alternative to isotropic Gaussian noise in Differentially Private Stochastic Gradient Descent (DP-SGD), aiming to preserve gradient directions and improve the utility-privacy trade-off. The authors propose using the von Mises-Fisher (VMF) distribution to perturb gradients based on angular distance, ensuring directional privacy while maintaining model performance.
- **Key Findings:**
  - Utility:

    - VMF noise preserves higher model accuracy compared to Gaussian noise at equivalent privacy budgets (ε). For example:

      - On CIFAR-10 with LeNet, VMF (ε=1) achieves 50.7% accuracy vs. Gaussian (ε=1) at 37.4%.

      - For MLPs, VMF often marginally outperforms non-private baselines due to noise acting as regularization.

  - Privacy Defense:

    - Membership Inference Attacks (MIA): VMF reduces attack success rates (AUC) closer to random chance (50%) compared to Gaussian noise. For example, on CIFAR-10 with LeNet, VMF (ε=50) yields 50.8% AUC vs. Gaussian (ε=50) at 51.7%.

    - Reconstruction Attacks: VMF introduces higher reconstruction error (MSE), making recovered data noisier. For example, on Fashion-MNIST with LeNet, VMF (ε=1) achieves 0.19 MSE vs. Gaussian (ε=1) at 0.01 MSE.

- **Conclusion:** Directional noise via VMF offers a superior utility-privacy trade-off compared to Gaussian noise in DP-SGD, validated through empirical attacks and theoretical guarantees. Future work includes optimizing computational costs for high-dimensional data and extending the approach to NLP tasks. The results advocate for directional mechanisms as a promising alternative in differentially private deep learning.

### Paper 5

### Title: Delving into Differentially Private Transformer

- **Citation:** Ding, Youlong, et al. "Delving into Differentially Private Transformer." arXiv preprint arXiv:2405.18194 (2024).
- **Author(s):** Youlong Ding, Xueyang Wu, Yining Meng, Yonggang Luo, Hao Wang, Weike Pan
- **Publication Year:** 2024
- **Summary:** This paper explores the challenges of training Transformer models with differential privacy, arguing that the problem can be "reduced" to the more understood problem of training vanilla neural nets with differential privacy. The authors propose two novel techniques, Re-Attention Mechanism and Phantom Clipping, to address the unique difficulties encountered in training DP Transformers.
The authors propose a modular approach to address these challenges, focusing on reducing the problem of training DP Transformers to the more basic problem of training DP vanilla neural nets.
- **Key Findings:**
  - Attention Distraction: The attention mechanism in Transformers can be distorted by DP noise, leading to suboptimal model accuracy.
  - Gradient Clipping: Existing techniques for efficient gradient clipping are not compatible with the standard Transformer architecture due to embedding sharing.
- **Eperimental Results:**
  - Re-Attention Mechanism: This mechanism mitigates attention distraction by tracking the variance of each layer's output distribution during forward propagation.
  The Re-Attention Mechanism consistently improves training stability and convergence, especially on imbalanced datasets.
  - Phantom Clipping: This technique enables efficient gradient clipping for Transformers with embedding sharing, generalizing existing methods.
  Phantom Clipping significantly improves memory efficiency and training speed compared to existing methods.

- **Conclusion:** The authors demonstrate that their modular approach effectively addresses the unique challenges of training DP Transformers. The proposed solutions, Re-Attention Mechanism and Phantom Clipping, contribute to advancing research in differentially private deep learning.

### Paper 6

### Title: Guiding Text-to-Text Privatization by Syntax

- **Citation:** Arnold, Stefan, Dilara Yesilbas, and Sven Weinzierl. "Guiding text-to-text privatization by syntax." arXiv preprint arXiv:2306.01471 (2023).
- **Author(s):** Stefan Arnold, Dilara Yesilbas, Sven Weinzierl
- **Publication Year:** 2023
- **Summary:** The paper discusses advancements in text-to-text privatization using Metric Differential Privacy, focusing on enhancing user privacy while maintaining the grammatical integrity of the text. Traditional methods of anonymizing text data often fail to preserve the syntactic roles of words, resulting in incoherent outputs primarily composed of nouns.
- **Key Findings:**
  - Text-to-text privatization methods based on metric differential privacy often fail to preserve the grammatical properties of words after substitution.
  - The MADLIB mechanism, which operates on word embeddings, tends to produce surrogate texts dominated by nouns.
  - The authors propose a modification to MADLIB that incorporates grammatical categories into the candidate selection process.
  - This modification significantly improves the preservation of grammatical categories in surrogate texts.
  - The authors demonstrate that their modified mechanism achieves better performance in downstream tasks while maintaining comparable privacy guarantees.
  - The paper highlights the potential of incorporating grammatical properties into text-to-text privatization to enhance the utility of surrogate texts.
  - The authors acknowledge the need for further research to address the information leakage associated with their modification.
- **Conclusion:** In conclusion, the research demonstrates that integrating grammatical properties into text-to-text privatization can significantly enhance both privacy and utility. While prior studies primarily focused on geometric properties, this work highlights the importance of linguistic features in maintaining the quality of privatized texts. Despite potential risks of information leakage, the authors argue that their method represents a step forward in balancing privacy concerns with the need for coherent, contextually relevant text outputs. Future work could explore refining the candidate selection process to further mitigate privacy risks.

### Paper 7

### Title: Large-Scale Differentially Private BERT

- **Citation:** Anil, Rohan, et al. "Large-scale differentially private BERT." arXiv preprint arXiv:2108.01624 (2021).
- **Author(s):** Rohan Anil, Badih Ghazi, Vineet Gupta, Ravi Kumar, Pasin Manurangsi
- **Publication Year:** 2021
- **Summary:** This paper explores the pretraining of BERT-Large using Differentially Private Stochastic Gradient Descent (DP-SGD). The authors address the challenges of training large language models with differential privacy, which is crucial for protecting user data privacy.
- **Key Findings:**
  - Negative Interaction of Naive DP-SGD with Scale-Invariant Layers: The authors highlight the importance of a large weight decay parameter in the Adam optimizer to handle scale-invariant layers, which are common in BERT-Large. This insight helps in tuning hyper-parameters effectively to achieve higher accuracy.

  - Mega-Batches Improve Accuracy: They demonstrate that scaling up batch sizes to around 2 million examples significantly improves the utility of DP-SGD for BERT-Large. This approach achieves a masked language model (MLM) accuracy of 60.5% at a batch size of 2 million, compared to the non-private BERT models' accuracy of around 70%.

  - Increasing Batch Size Schedule: The authors propose an increasing batch size schedule that improves training efficiency while maintaining accuracy. This schedule is motivated by the observation that the gradient signal-to-noise ratio (gradient-SNR) decreases over time, and increasing the batch size helps mitigate this issue.
- **Conclusion:** Overall, the paper establishes a high-accuracy baseline for DP BERT-Large pretraining and provides valuable insights into the practical challenges and solutions for training large language models with differential privacy.

### Paper 8

### Title: Large Scale Private Learning via Low-rank Reparametrization

- **Citation:** Yu, Da, et al. "Large scale private learning via low-rank reparametrization." International Conference on Machine Learning. PMLR, 2021.
- **Author(s):** Da Yu, Huishuai Zhang, Wei Chen, Jian Yin, Tie-Yan Liu
- **Publication Year:** 2021
- **Summary:** The paper introduces a novel method for private learning, focusing on Reparametrized Gradient Perturbation (RGP) to enhance differential privacy while minimizing memory usage.
- **Key Findings:** RGP employs a reparametrization of weight matrices, utilizing gradient-carrier matrices to streamline gradient calculations and making use of historical gradients for optimization.
- **Eperimental Results:** The proposed method demonstrates significant performance improvements on benchmark tests, achieving notable accuracy while effectively managing privacy budgets and surpassing existing baseline methods.
- **Conclusion:** RGP successfully balances privacy with model performance. Future directions include exploring more applications and optimizing the method for varied models, thus lowering the risks associated with privacy without sacrificing effectiveness.

### Paper 9

### Title: Learning and Evaluating a Differentially Private Pre-trained Language Model

- **Citation:** Shlomo Hoory, Amir Feder, Avichai Tendler, Sofia Erell, Alon Peled-Cohen, Itay Laish, Hootan Nakhost, Uri Stemmer, Ayelet Benjamini, Avinatan Hassidim, and Yossi Matias. 2021. Learning and Evaluating a Differentially Private Pre-trained Language Model. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 1178–1189, Punta Cana, Dominican Republic. Association for Computational Linguistics.
- **Author(s):** Shlomo Hoory, Amir Feder, Avichai Tendler, Sofia Erell, Alon Peled-Cohen, Itay Laish, Hootan Nakhost, Uri Stemmer, Ayelet Benjamini, Avinatan Hassidim, Yossi Matias
- **Publication Year:** 2021
- **Summary:** This paper addresses the challenge of training large language models (LLMs) like BERT on sensitive data while preserving user privacy. The authors propose a novel approach to train a differentially private BERT model on medical data, specifically clinical notes, with a strong privacy guarantee of  = 1.1.
- **Key Findings:**
  - Differentially Private WordPiece Algorithm: The authors introduce a differentially private version of the WordPiece algorithm, which allows for the creation of domain-specific vocabularies while maintaining privacy.
  - Training a Differentially Private BERT Model: The paper presents a method for training a differentially private BERT model using the DP-SGD algorithm, overcoming challenges related to large batch sizes and parallelism.
  - Privacy Evaluation: The authors employ a secret sharer membership test to demonstrate that the trained DP-BERT model does not memorize private information.
- **Eperimental Results:**
  - The DP-BERT model achieves comparable performance to non-private BERT models on entity extraction tasks from clinical notes, demonstrating that privacy-preserving training does not significantly degrade model utility.
  - The secret sharer test confirms that the DP-BERT model effectively prevents information leakage, ensuring that private information is not memorized.
- **Conclusion:** This work provides a practical solution for training LLMs on sensitive data while maintaining privacy guarantees. The proposed approach offers a balance between model performance and privacy, paving the way for the development of more secure and responsible NLP applications.

### Paper 10

### Title: Natural Language Understanding with Privacy-Preserving BERT

- **Citation:** Qu, Chen, et al. "Natural language understanding with privacy-preserving bert." Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021.
- **Author(s):** Chen Qu, Weize Kong, Liu Yang, Mingyang Zhang, Michael Bendersky, Marc Najork
- **Publication Year:** 2021
- **Summary:** This paper addresses the challenge of preserving privacy in Natural Language Understanding (NLU) applications using pretrained Language Models (LMs) like BERT. It focuses on the Local Privacy setting, where users privatize their data before sending it to service providers.
- **Key Findings:**
  - Investigates the privacy and utility implications of applying 𝑑𝜒-privacy to BERT fine-tuning.
  - Proposes privacy-adaptive LM pretraining methods to improve BERT's utility on privatized text.
  - Quantifies the level of privacy preservation and provides guidance on privacy configuration.
- **Methods:**
  - Token Representation Privatization: Perturbs token embeddings with noise drawn from a specific distribution.
  - Text-to-Text Privatization: Replaces words in the original text with semantically similar words using nearest neighbor search in the embedding space.
  - Privacy-Adaptive Pretraining: Trains BERT on large-scale public text corpora that have been privatized using different variants of the Masked Language Model (MLM) loss.
- **Conclusion:**
  - Text-to-text privatization outperforms token representation privatization in fine-tuning due to the effectiveness of nearest neighbor search.
  - Privacy-adaptive pretraining significantly improves BERT's performance on privatized text, making it more robust and effective.
  - Denoising MLM, which predicts the original tokens, is the most effective privacy-adaptive pretraining method.
  - Token representation privatization outperforms text-to-text privatization when noise is large with privacy-adaptive pretraining.

### Paper 11

### Title: Privacy- and Utility-Preserving Textual Analysis via Calibrated Multivariate Perturbations

- **Citation:** Feyisetan, Oluwaseyi, et al. "Privacy-and utility-preserving textual analysis via calibrated multivariate perturbations." Proceedings of the 13th international conference on web search and data mining. 2020.
- **Author(s):** Oluwaseyi Feyisetan, Borja Balle, Thomas Drake, Tom Diethe
- **Publication Year:** 2020
- **Summary:** This paper presents a method for privacy-preserving text perturbation using a technique called dχ-privacy. The method aims to protect user privacy while maintaining the utility of the data for downstream machine learning tasks.
- **Key Findings:**
  - Scalable Mechanism: The paper proposes a scalable mechanism for text analysis that satisfies dχ-privacy. This mechanism operates on individual words, perturbing them by adding noise to their vector representations in a high-dimensional space defined by word embedding models.
  - Privacy Proof: The paper provides a formal privacy proof demonstrating that the mechanism satisfies dχ-privacy. The privacy parameter ε controls the strength of the privacy guarantee, with larger values of ε providing stronger privacy guarantees.
  - Privacy Calibration: The paper presents a methodology for calibrating the privacy parameter ε based on the geometric structure of the word embedding model. This involves analyzing plausible deniability statistics, such as the probability of not modifying the input word and the effective support of the output distribution.
  - Utility Experiments: The paper conducts utility experiments on three datasets (IMDb movie reviews, Enron emails, and InsuranceQA) to demonstrate the trade-off between privacy and utility for different values of ε. The results show that the mechanism can achieve practical utility with minimal loss in performance for tasks like binary sentiment analysis.
  - Privacy Audit Experiments: The paper compares the privacy guarantees of the proposed mechanism against two baseline query scrambling methods (Versatile and Incognito). The results demonstrate that the proposed mechanism provides significantly better privacy guarantees, preventing attacks that successfully identify perturbed queries in the baseline methods.
- **Conclusion:** Overall, the paper presents a novel and effective approach for privacy-preserving text analysis that balances privacy guarantees with utility for downstream machine learning tasks.

### Paper 12

### Title: Privacy-Preserving Parameter-Efficient Fine-Tuning for Large Language Model Services

- **Citation:** Li, Yansong, Zhixing Tan, and Yang Liu. "Privacy-preserving prompt tuning for large language model services." arXiv preprint arXiv:2305.06212 (2023).
- **Author(s):** Yansong Li, Zhixing Tan, Yang Liu
- **Publication Year:** 2023
- **Summary:** This paper proposes a framework called RAPT for customizing Large Language Models (LLMs) with private data while ensuring privacy. RAPT addresses the challenges of data leakage and performance degradation associated with traditional Parameter-Efficient Fine-Tuning (PEFT) methods.
- **Key Findings of RAPT:**
  - Local Privacy: RAPT utilizes a local privacy approach, allowing users to privatize their data locally using a text-to-text local differential privacy mechanism.
  - Part-of-Speech Constrained Text-to-Text Privatization (PCT2T): This technique improves the grammatical coherence and semantic similarity of privatized text by restricting word substitutions within their original Part-of-Speech (POS) roles.
  - Privatized Token Reconstruction Task: RAPT introduces a novel task that is trained jointly with the downstream task, enabling LLMs to learn better task-dependent representations and mitigate performance degradation caused by privatization.
- **Eperimental Evaluation:**
  - Privacy Experiments: Simulated privacy attacks, including embedding inversion, attribute inference, and NER attacks, demonstrate RAPT's effectiveness in protecting privacy.
  - Utility Experiments: Experiments on various NLU and NLG tasks show that RAPT achieves competitive performance compared to other PEFT methods and privacy-preserving techniques, such as CAPE and DPNR.
- **Conclusion:** RAPT offers a practical and effective solution for customizing LLM services with private data while providing robust privacy protection. The framework combines the benefits of local privacy, PCT2T, and privatized token reconstruction to achieve a favorable privacy-utility trade-off.

### Paper 13

### Title: Privacy-Preserving Text Classification on BERT Embeddings with Homomorphic Encryption

- **Citation:** Lee, Garam, et al. "Privacy-preserving text classification on BERT embeddings with homomorphic encryption." arXiv preprint arXiv:2210.02574 (2022).
- **Author(s):** Garam Lee, Minsoo Kim, Jai Hyun Park, Seung-won Hwang, Jung Hee Cheon
- **Publication Year:** 2022
- **Summary:** This paper proposes a method for privacy-preserving text classification using homomorphic encryption (HE) and BERT embeddings. The method addresses the privacy concerns associated with traditional text embedding models, which can leak sensitive information about the text.
- **Key Findings:**
  - BERT Embedding-based Method: The proposed method utilizes pre-trained BERT embeddings, which are semantically rich and significantly outperform previous methods like PrivFT, which relied on fastText embeddings.
  - Efficient GPU Implementation: The paper presents an efficient GPU implementation of the CKKS encryption scheme, which is well-suited for real-valued data like BERT embeddings. This implementation improves computation speed and allows for higher security levels compared to previous methods.
  - Encrypted Logistic Regression: The method employs an encrypted logistic regression model as the downstream classifier, enabling training and inference on encrypted data.
- **Advantages:**
  - Privacy Protection: The use of HE ensures that all computations are performed on encrypted data, preventing any leakage of sensitive information.
  - Utility Preservation: The method achieves high accuracy on downstream text classification tasks, demonstrating that HE can preserve the utility of BERT embeddings.
  - Efficiency: The GPU implementation significantly improves training efficiency compared to previous methods.
- **Conclusion:** The proposed method provides a practical and efficient solution for privacy-preserving text classification, leveraging the power of BERT embeddings and the security of homomorphic encryption. While further research is needed to address the computational cost of HE-based models, this work demonstrates the potential of HE for protecting privacy in NLP applications.

### Paper 14

### Title: Privformer: Privacy-preserving Transformer with MPC

- **Citation:** Y. Akimoto, K. Fukuchi, Y. Akimoto and J. Sakuma, "Privformer: Privacy-preserving Transformer with MPC," 2023 IEEE 8th European Symposium on Security and Privacy (EuroS&P), Delft, Netherlands, 2023, pp. 392-410, doi: 10.1109/EuroSP57164.2023.00031.
- **Author(s):** Yoshimasa Akimoto, Kazuto Fukuchi, Youhei Akimoto, Jun Sakuma
- **Publication Year:** 2023
- **Summary:** This paper presents a 3-party Multi-Party Computation (MPC) protocol for secure inference of the Transformer in the honest majority setting. The protocol aims to enable the use of Transformer models deployed on cloud servers through APIs while preserving both model and user privacy.
- **Key Findings:**
  - Efficient MPC protocol for attention: The paper introduces ReLU attention, a variant of the attention mechanism that replaces the sigmoid function with the ReLU function, making it more MPC-friendly. This reduces the number of exponentiation calls from O(S²) to O(S), where S is the sequence length.
  - Protocols for masked attention: Two protocols for masked attention are proposed: one with O(S²) time complexity in O(1) rounds and another with O(S) time complexity in O(S) rounds. The choice of protocol depends on the communication environment.
  - Novel MPC protocol for square root inverse: A new protocol for square root inverse is introduced that does not leak any intermediate values, ensuring security in the honest majority setting.
- **Eperimental Results:**
  - The proposed protocols were implemented on top of the FALCON framework and evaluated in LAN and WAN environments.
  - ReLU attention significantly outperforms softmax attention in terms of computation time and communication bandwidth, especially for longer sequences.
  - Masked ReLU attentionQK achieves the best performance for most sequence lengths, while masked ReLU attentionVK is expected to be more efficient for even longer sequences.
  - The entire Privformer protocol can infer a 64-word sentence in about 19 minutes in a LAN environment

### Paper 15

### Title: PrivLM-Bench: A Multi-level Privacy Evaluation Benchmark for Language Models

- **Citation:** Li, Haoran, et al. "P-bench: A multi-level privacy evaluation benchmark for language models." arXiv preprint arXiv:2311.04044 (2023).
- **Author(s):** Haoran Li, Dadi Guo, Donghao Li, Wei Fan, Qi Hu, Xin Liu, Chunkit Chan, Duanyi Yao, Yuan Yao, Yangqiu Song
- **Publication Year:** 2023
- **Summary:** This paper introduces PrivLM-Bench, a benchmark designed to evaluate the privacy-utility trade-off of privacy-preserving language models (PPLMs). PrivLM-Bench addresses the limitations of existing evaluation methods that rely solely on differential privacy (DP) parameters.
- **Key Findings:**
  - Multi-faceted Privacy Objectives: PrivLM-Bench defines privacy objectives at both the fine-tuning and inference stages, highlighting the importance of protecting inference data privacy.
  - Unified Pipeline: It provides a unified pipeline for evaluating PPLMs, enabling fair comparisons among different implementations.
  - Empirical Privacy Attacks: PrivLM-Bench incorporates various privacy attacks, including data extraction attacks, membership inference attacks, and embedding-level attacks, to empirically quantify privacy leakage
- **Eperimental Results:**
  - Utility Degradation: DP tuning leads to utility degradation, with fine-tuning generally yielding the best results but suffering from significant accuracy drops in prompt and prefix tuning.
  - Fine-tuning Data Privacy: DP-tuned models effectively protect fine-tuning data privacy against membership inference attacks.
  - Inference Data Privacy: Naive DP tuning fails to protect inference data privacy, highlighting the need for additional mechanisms.
  - Attack Effectiveness: Empirical privacy attacks are less potent than anticipated, suggesting a gap between defense mechanisms' assumed capabilities and actual attack performance.
- **Potential Applications:**
  - Comparison of PPLMs: PrivLM-Bench enables fair comparisons among different PPLMs.
  - Development of New Attacks and Defenses: It provides a platform for developing new attacks and defenses.
  - Verification of PPLM Implementations: It offers an empirical evaluation to verify the correctness of PPLM implementations.
- **Conclusion:**

### Paper 16

### Title: Split-and-Denoise: Protect Large Language Model Inference with Local Differential Privacy

- **Citation:** Mai, Peihua, et al. "Split-and-Denoise: Protect large language model inference with local differential privacy." arXiv preprint arXiv:2310.09130 (2023).
- **Author(s):** Peihua Mai, Ran Yan, Zhe Huang, Youjia Yang, Yan Pang
- **Publication Year:** 2023
- **Summary:** This paper introduces Split-N-Denoise (SnD), a framework designed to protect user privacy during large language model (LLM) inference. SnD addresses the privacy concerns associated with transmitting sensitive information to servers for LLM processing.
- **Key Findings:** Split Inference: The LLM is split, with the token embedding layer executed on the client side, minimizing computational overhead.
Local Differential Privacy (LDP): Clients introduce noise to their token embeddings before transmitting them to the server, ensuring privacy.
Denoising: A pre-trained denoising model is deployed on the client side to enhance the utility of the noisy embeddings received from the server.
- **Eperimental Results:**
  - Privacy Evaluation: SnD demonstrates strong privacy guarantees, with negligible attack success rates and low mutual information under various privacy budgets.
  - Downstream Task Performance: SnD maintains acceptable accuracy on various downstream tasks, including sentence classification, pair similarity, and textual entailment.
  - Comparison with Baselines: SnD consistently outperforms existing privacy-preserving LLM inference approaches, including TokenEmbPriv, Text2Text, and RAPT.
- **Advantages of SnD:**
  - Improved Privacy-Utility Tradeoff: SnD outperforms existing DP-based baselines by over 10% on average, offering clients a privacy-preserving solution for local privacy protection.
  - Minimal Model Modifications: SnD requires no modifications to the LLM parameters, making it suitable for black-box API access.
  - Effective Denoising: The client-side denoising model leverages knowledge of noise levels and raw inputs to enhance embedding utility.
- **Conclusion:** SnD offers a promising solution for privacy-preserving LLM inference, balancing privacy protection with utility enhancement. Its effectiveness in optimizing the privacy-utility tradeoff makes it a valuable contribution to the field of trustworthy machine learning.

- **Future Work:**
  - Scalability to Larger LLMs: Extending SnD to larger LLMs requires more sophisticated noise and denoising mechanisms.
  - Reducing User Computation Cost: Designing lightweight denoising mechanisms to minimize user computation overhead is crucial.
  - Sequence-to-Sequence (S2S) Inference: Generalizing SnD to S2S inference models requires addressing noise amplification issues.

### Paper 17

### Title:

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

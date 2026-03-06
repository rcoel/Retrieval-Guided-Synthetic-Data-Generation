# Retrieval Guided Synthetic Data Generation

**Authors:**

- Dr Rajashree Shettar (<rajashreeshettar@rvce.edu.in>)
- Reevan Coelho (<reevancoelho.scs24@rvce.edu.in>)
- Veerasagar S S (<veerasagarss.scs24@rvce.edu.in>)

**Affiliation:** Department of Computer Science and Engineering, RV College of Engineering, Bangalore, India

## Abstract

Retrieval-augmented synthetic data generation has emerged as an effective approach for augmenting private-domain corpora while preserving data confidentiality. Our framework employs LoRA-based parameter-efficient fine-tuning of a Qwen 2.5–7B Instruct model on private examples, supplemented by a large-scale k-nearest-neighbor search over a globally available public dataset. Each private instance is paired with the top-k semantically related chunks from the public corpus, and these retrieved passages serve as prompts for the Qwen model to generate synthetic variants. The assembled synthetic dataset is used to fine-tune a standard BERT-base classifier. Experimental results on SST-2 demonstrate that the BERT model trained solely on our synthetic data achieves 94.0 % accuracy, rivaling fully supervised baselines. Additional evaluations on GLUE benchmarks yield competitive performance: MNLI at 87.2 %, QQP at 91.0 %, QNLI at 89.6 %, and CoLA at 54.3 Matthews correlation. These findings indicate that our retrieval-augmented pipeline can substantially reduce reliance on human annotation without sacrificing task performance.

**Keywords:** Synthetic data generation, retrieval augmentation, PEFT, Qwen Instruct, LoRA, BERT, GLUE benchmarks.

## 1. Introduction

Data-driven natural language processing (NLP) systems have demonstrated remarkable performance across tasks such as sentiment analysis, question answering, and natural language inference. These advances, however, rely heavily on large volumes of high-quality annotated data, whose collection can be expensive, time-consuming, or legally restricted when sensitive information is involved. In domains such as healthcare, finance, and personal communication, privacy concerns often preclude the sharing of original data, creating a barrier to developing robust models.

Synthetic data generation using Large Language Models (LLMs) offers a viable solution by creating artificial examples that mimic private-domain distributions without exposing raw content. Yet purely generative approaches can overfit to limited private examples or produce outputs that lack contextual relevance. To address these shortcomings, retrieval-augmented generation incorporates semantically related content drawn from a broad public corpus, guiding the model toward richer, more varied outputs while maintaining domain fidelity.

Parameter-efficient fine-tuning (PEFT) techniques, such as Low-Rank Adaptation (LoRA), enable the adaptation of large-scale LLMs to specialized domains without the computational overhead of full model retraining. By injecting a small set of trainable parameters into the attention layers, LoRA dramatically reduces memory usage and training time, making it practical to fine-tune multi-billion-parameter models on private data within constrained environments.

In this work, as shown in Figure 1, we propose a retrieval-augmented [3] synthetic data generation framework built around a LoRA-adapted Qwen 2.5–7B Instruct model [4]. Our pipeline proceeds in three key stages. First, a globally available public dataset is segmented into fixed-size chunks and indexed for semantic similarity. Second, for each private-domain example, we retrieve the top-k most relevant public passages via k-nearest-neighbor search and concatenate them with the private prompt. Figure 1 shows the overall methodology adopted for Synthetic corpus generation.

> **Figure 1.** The Method Overview (Placeholder for image)

To validate the effectiveness of our approach, we conduct extensive experiments on the Stanford Sentiment Treebank (SST-2) and four additional tasks from the GLUE[24] benchmark suite: MNLI, QQP, QNLI, and CoLA. On SST-2, a BERT-base[21] classifier trained exclusively on our synthetic corpus achieves 94.0 % accuracy, closely matching fully supervised baselines. Comparable gains across the other tasks demonstrate that our method generalizes beyond sentiment analysis, substantially reducing the need for human annotation.

The main contributions of this paper are:
- A novel retrieval-augmented synthetic data generation framework that leverages PEFT-LoRA to adapt a large Instruct model to private data;
- A scalable chunking and k-NN retrieval mechanism over public corpora to guide synthetic example creation;
- A comprehensive evaluation showing that synthetic-only training achieves near-state-of-the-art performance on diverse NLP tasks;
- Public release of our codebase and generated corpora to support reproducibility and foster further research in privacy-preserving data augmentation.

The remainder of this paper is organized as follows. Section II reviews related work on synthetic data generation, retrieval-augmented methods, and parameter-efficient fine-tuning. Section III details our Methodology. Section IV describes the experimental setup. Section V presents results and analysis. Section VI concludes with discussion and future directions.

## 2. Literature Review

The increasing deployment of large language models (LLMs) in various applications has raised significant concerns about the privacy of both training data and user inputs during inference. This section reviews recent research addressing these privacy challenges, focusing on techniques applied to both training and inference stages. A dominant theme is the application of Differential Privacy (DP) and its variants.

### 2.1 Differential Privacy and its Adaptations

Several studies investigate the use of standard Differentially Private Stochastic Gradient Descent (DP-SGD) for training LMs. Anil et al. apply DP-SGD to pre-training BERT-Large, highlighting the critical role of hyperparameter tuning and large batch sizes, and demonstrate the trade-off between the privacy parameter ε and downstream utility[1]. Hoory et al. further evaluate DP-SGD in the context of both pre-training and fine-tuning, confirming that naïve application of DP can significantly degrade model performance without careful calibration[2]. To mitigate these issues, adaptive DP methods have been proposed. Wu et al. introduce Adaptive Differential Privacy (ADP), which dynamically adjusts noise levels based on an estimated privacy risk derived from model perplexity, yielding improved utility relative to standard DP-SGD[4]. Faustini et al. propose directional noise drawn from a von Mises–Fisher distribution, preserving gradient directionality better than isotropic Gaussian noise and enhancing the utility-privacy trade-off[5]. For memory-efficient private learning at scale, Yu et al. present Reparameterized Gradient Perturbation (RGP), which minimizes additional storage overhead[6]. Recently, Ding et al. address DP in Transformer architectures by introducing "Re-Attention" to reduce attention distraction and "Phantom Clipping" for efficient gradient clipping in DP-Transformers[3]. Yue et al. demonstrate that DP methods can also generate high-utility synthetic text for customer feedback applications[7].

### 2.2 Local Differential Privacy and Input Perturbation

Another line of work shifts privacy protection to the user side via Local Differential Privacy (LDP). Qu et al. apply dχ-privacy to BERT fine-tuning and propose privacy-adaptive pre-training to enhance robustness on privatized text inputs[8]. Feyisetan et al. similarly employ dχ-privacy by adding calibrated multivariate noise to word embeddings, achieving privacy-preserving textual analysis with minimal utility loss[9]. Li et al. introduce RAPT, combining LDP with Part-of-Speech Constrained Text-to-Text Privatization (PCT2T) and a privatized token reconstruction task to jointly optimize privacy and utility[10]. The Split-and-Denoise (SnD) framework by Mai et al. partitions the LLM, applying LDP to client-side embeddings and using a denoiser on the server to mitigate noise effects[11]. These approaches demonstrate that client-side perturbation can yield favorable utility-privacy trade-offs by offloading privacy mechanisms from the server.

### 2.3 Model Splitting and Federated Learning

Model splitting techniques seek to protect both model parameters and user data by dividing computation across client and server. Shen et al. propose the Split-and-Privatize (SAP) framework, which splits model layers between client and server and applies token-level privatization to intermediate activations, with Contributing-Token-Identification (CTI) reducing noise on critical tokens[12]. Huang et al. introduce TextHide for federated learning, encrypting user data representations before aggregation to prevent leakage in cross-device updates[13]. These methods balance privacy, performance, and computational overhead in distributed settings.

### 2.4 Alternative Privacy-Enhancing Technologies

Beyond DP, other cryptographic and MPC-based solutions have been explored. Lee et al. demonstrate that homomorphic encryption (HE) can be combined with BERT embeddings to enable privacy-preserving text classification without decrypting user inputs on the server[14]. Akimoto et al. present Privformer, a three-party Multi-Party Computation (MPC) protocol tailored for secure Transformer inference, featuring ReLU attention and specialized protocols for masked attention operations[15].

### 2.5 Privacy-Preserving Inference

Specific techniques target the inference stage to prevent input reconstruction. Zhou et al. propose TextFusion, which fuses multiple tokens into composite representations, breaking the one-to-one mapping between embeddings and raw words[16]. Zhou et al. later introduce TextMixer, which mixes a target input with auxiliary texts using a multi-input multi-output (MIMO) network to obfuscate sensitive content during inference[17]. These methods offer practical defenses against inference-time privacy attacks.

## 3. Methodology

The core of our retrieval-augmented synthetic data generation framework lies in three intertwined components: semantic indexing of a public corpus, parameter-efficient fine-tuning of an LLM with LoRA adapters, and synthetic example synthesis. We first formalize the retrieval process. Let $\mathcal{C}$ denote the set of public-corpus chunks, each mapped to a d-dimensional embedding via a sentence encoder $f_\text{enc}$. For a private example $x_i$, we compute its query embedding

$$ q_i = f_\text{enc}(x_i) $$

We then retrieve the top-$K$ passages by solving

$$ \{p_{i,j}\}_{j=1}^K = \text{arg max}_{p \in \mathcal{C}} \text{ sim}(q_i, f_\text{enc}(p)) $$

where $\text{sim}$ denotes cosine similarity. The retrieval-augmented prompt $r_i$ is constructed by concatenating $x_i$ with the retrieved passages using a special delimiter token.

To adapt the Qwen 2.5–7B Instruct model while preserving privacy, we integrate LoRA adapters into each multi-head attention layer. Specifically, for a weight matrix $W$ in the query or value projection, we replace

$$ h = W x $$

with

$$ h = W x + B A x $$

where $A$ and $B$ are trainable low-rank matrices of rank $r$. Only $A$ and $B$ are updated during fine-tuning, reducing the parameter footprint. The model is trained to minimize the cross-entropy loss over next-token prediction:

$$ \mathcal{L}_\text{CE} = - \sum \log P_\theta (x_t \mid x_{<t}, r_i) $$

where $\theta$ denotes the union of all LoRA parameters.

Once fine-tuning converges, synthetic examples $\hat{y}_i$ are generated by sampling from $P_\theta$ using nucleus sampling with parameter $p$ and temperature $\tau$. We enforce a fluency constraint by computing the perplexity:

$$ \text{PPL}(\hat{y}_i) = \exp \left( - \frac{1}{|\hat{y}_i|} \sum \log P_\theta (\hat{y}_{i,t} \mid \hat{y}_{i,<t}, r_i) \right) $$

and discarding any $\hat{y}_i$ with $\text{PPL}(\hat{y}_i) > \eta$. Label transfer is trivial for classification tasks: the synthetic example inherits the original label $y_i$.

The assembled synthetic corpus $\mathcal{D}_\text{syn}$ is then used to fine-tune a downstream BERT-base classifier. The classifier parameters $\phi$ are optimized via:

$$ \min_\phi \sum_{(\hat{x}, y) \in \mathcal{D}_\text{syn}} \mathcal{L}_\text{cls} (f_\phi(\hat{x}), y) $$

where $\mathcal{L}_\text{cls}$ is the classification loss over the number of classes.

Algorithm 1 outlines the complete retrieval-augmented synthetic data generation procedure.

### Algorithm 1: Retrieval-Augmented Synthetic Data Generation

**Input:** Private dataset $\mathcal{D}_\text{priv} = \{(x_i, y_i)\}_{i=1}^N$, public corpus $\mathcal{C}$
**Output:** Synthetic corpus $\mathcal{D}_\text{syn}$

1. Build semantic index on chunks $\mathcal{C}$ using embeddings $f_\text{enc}(\cdot)$.
2. **for each** $(x_i, y_i)$ in $\mathcal{D}_\text{priv}$ **do**
3.     $q_i \leftarrow f_\text{enc}(x_i)$
4.     $\{p_{i,j}\}_{j=1}^K \leftarrow \text{top-K retrieval from index by } \text{sim}(q_i, \cdot)$
5.     $r_i \leftarrow \text{concatenate}(x_i, p_{i,1}, \dots, p_{i,K})$
6. **end for**
7. Fine-tune Qwen model with LoRA adapters on $\{(r_i, x_i)\}$ to minimize $\mathcal{L}_\text{CE}$.
8. **for each** $r_i$ **do**
9.     Sample $\hat{y}_i \sim P_\theta(\cdot \mid r_i)$ with nucleus sampling $(p, \tau)$
10. **if** $\text{PPL}(\hat{y}_i) \leq \eta$ **then**
11.        add $(\hat{y}_i, y_i)$ to $\mathcal{D}_\text{syn}$
12. **end if**
13. **end for**
14. **return** $\mathcal{D}_\text{syn}$

In implementation, embedding extraction and index construction utilize FAISS[27] with GPU acceleration, while private fine-tuning and generation employ unsloth[28] training strategy. This methodology ensures secure handling of private data, efficient retrieval operations, and high-quality synthetic data tailored for downstream model training.

## 4. Experimental Setup and Datasets

To comprehensively evaluate our retrieval-augmented synthetic data pipeline, we conduct experiments on five benchmark datasets spanning sentiment analysis, natural language inference, paraphrase detection, and acceptability judgment. All private training instances are drawn from the original training splits, while held-out validation and test splits remain untouched for fair evaluation.

### 4.1 Datasets

The Stanford Sentiment Treebank (SST-2) comprises movie review sentences labeled positive or negative. We use the standard train/validation/test split of 67,349/872/1,821 examples. For natural language inference, we include the Multi-Genre Natural Language Inference (MNLI) corpus, containing approximately 393,000 premise–hypothesis pairs across diverse genres with matched and mismatched test sets of 9,815 and 9,832 instances, respectively. Paraphrase detection is evaluated on the Quora Question Pairs (QQP) dataset, which offers 363,846 training pairs and 40,431 validation instances annotated for semantic equivalence. Question-answering natural language inference (QNLI) is derived from the Stanford Question Answering Dataset, reformulated to an NLI task with 104,744 training and 5,463 validation examples. Finally, the Corpus of Linguistic Acceptability (CoLA) contains 8,551 sentences labeled grammatically acceptable or not, with a validation set of 1,043 examples.

### 4.2 Retrieval Index and Public Corpus

Our public corpus consists of a 6.41 million-document, 20231101.en subset of Wikipedia dataset from wikimedia[25]. Documents are segmented into 256-token passages with 50% overlap, yielding approximately 200 million chunks. Embeddings are computed via a pre-trained Sentence-Transformer all-MPNet-base-v2[26] model producing 768-dimensional vectors. An FAISS-GPU index is built on a single NVIDIA T4 GPU. We fix K=5 for top-K retrieval, balancing contextual diversity and computational efficiency.

### 4.3 LoRA Fine-Tuning and Generation Hyperparameters

Parameter-efficient fine-tuning of the Qwen 2.5–7B Instruct model employs LoRA adapters of rank r = 8 in all attention projections. Training uses private prompts and retrieval-augmented contexts over 10,000 gradient steps, with a batch size of 64. We adopt a learning-rate schedule with 500-step linear warmup from 0 to 1e-4, followed by cosine decay to 1e-6. Weight decay of 0.01 is applied solely to LoRA parameters. Synthetic generations are sampled with nucleus sampling (p = 0.9) and temperature $\tau = 0.8$, with maximum length 128 tokens.

### 4.4 Downstream Classifier Training

The synthetic corpus for each task is used to train a separate BERT-base classifier. Inputs are tokenized to maximum sequence length 128, and classifiers are fine-tuned with AdamW for three epochs, a learning rate of 2e-5, and batch size 32. Model is then evaluated using the Huggingface evaluate library on GLUE benchmark.

### 4.5 Infrastructure and Reproducibility

All experiments run on NVIDIA T4 GPUs. Qwen 2.5-7B-Instruct is loaded in 4bit quantization for reduced memory usage using unsloth training framework. Bert model is trained via Huggingface trainer API.

## 5. Results and Analysis

**Table 1: Performance of BERT-base classifiers trained on synthetic data generated by different methods**

| Task | Metric | Yue et al.[7] | Arnold et al.[20] | **Ours** |
| :--- | :--- | :--- | :--- | :--- |
| **SST-2** | Accuracy | 88.5% | 90.4% | **94.0%** |
| **MNLI** | Accuracy | 82.3% | 84.5% | **87.2%** |
| **QQP** | F1 Score | 85.0% | 88.9% | **91.0%** |
| **QNLI** | Accuracy | 86.1% | 87.5% | **89.6%** |
| **CoLA** | Mathews Correlation | 48.0 | 50.6 | **54.3** |

Table 1 compares the downstream performance of BERT-base classifiers trained exclusively on synthetic data generated by two prior methods and our retrieval-augmented LoRA pipeline. On SST-2, the model trained on Yue et al.’s differentially private synthetic data[7] attains 88.5 % accuracy, while Arnold et al.’s syntax-guided privatization approach achieves 90.4%[20]. In contrast, our retrieval-augmented LoRA method yields 94.0 %, representing absolute gains of 5.5 % and 3.6 % over the two baselines, respectively.

For the MNLI task, Yue et al.’s pipeline reaches 82.3 % on the matched set and 81.7 % on the mismatched set, whereas Arnold et al. report 84.5 % and 83.9 % respectively. Our method improves these figures to 87.2 % (matched) and 86.5 % (mismatched), narrowing the gap with fully supervised performance. On QQP, prior synthetic-only training yields an F₁ score of 85.0 % for Yue et al. and 88.9 % for Arnold et al., while our approach achieves 91.0 %, a 2.1 % gain over the best baseline.

The QNLI results follow a similar trend: our 89.6 % accuracy exceeds Yue et al.’s 86.1 % and Arnold et al.’s 87.5 % by substantial margins. Finally, on the CoLA acceptability task, our Matthews correlation of 54.3 outperforms Yue et al.’s 48.0 and Arnold et al.’s 50.6. These improvements across five diverse benchmarks demonstrate that retrieval-guided prompts significantly enhance the utility of synthetic corpora compared to both differentially private generation and syntax-guided privatization.

Beyond absolute metrics, ablation experiments highlight the contributions of each component. Removing retrieval augmentation reduces SST-2 accuracy from 94.0% to 89.8%, while disabling LoRA fine-tuning destabilizes training and lowers performance to 91.1%. These results confirm that semantically informed retrieval and parameter-efficient adaptation are both essential for high-quality synthetic generation. Qualitative analysis of synthetic SST-2 examples reveals average model perplexities of 12.4 close to 10.2 on real data and a type token ratio of 0.084, indicating that our pipeline maintains both fluency and lexical diversity.

## 6. Conclusion and Future Work

We have presented a retrieval-augmented synthetic data generation framework that integrates k-NN retrieval over a large public corpus with LoRA-based fine-tuning of a Qwen 2.5–7B Instruct model on private examples. Experimental results on five GLUE benchmarks demonstrate that classifiers trained solely on our synthetic data outperform previous synthetic-only baselines by 3–6 % across tasks, achieving performance within 1–2 % of fully supervised models. Ablation studies and linguistic analyses corroborate the critical roles of retrieval guidance and parameter-efficient fine-tuning in producing high-fidelity synthetic examples.

Future work will explore automated hyperparameter optimization for chunk size, retrieval depth, and perplexity thresholds to further enhance generality across domains. Incorporating controlled generation objectives—such as style or sentiment intensity constraints—may enable the synthesis of task-tailored corpora. We also plan to extend our approach to multilingual and multimodal settings, and to integrate formal privacy auditing frameworks that provide provable guarantees. By releasing our codebase and synthetic corpora, we aim to foster continued advances in low-cost, privacy-preserving data augmentation.

## References

1. Anil, Rohan, et al. "Large-scale differentially private BERT." arXiv preprint arXiv:2108.01624 (2021).
2. Hoory, Shlomo, et al. "Learning and Evaluating a Differentially Private Pre-trained Language Model." Findings of the Association for Computational Linguistics: EMNLP 2021. (2021).
3. Lewis, Patrick, et al. "Retrieval-augmented generation for knowledge-intensive nlp tasks." Advances in neural information processing systems 33 (2020): 9459-9474.
4. Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers). 2019.
5. Ding, Youlong, et al. "Delving into Differentially Private Transformer." arXiv preprint arXiv:2405.18194 (2024).
6. Wu, Xinwei, Li Gong, and Deyi Xiong. "Adaptive Differential Privacy for Language Model Training." Proceedings of the First Workshop on Federated Learning for Natural Language Processing (FL4NLP 2022). (2022).
7. Faustini, Pedro, et al. "Directional Privacy for Deep Learning." arXiv preprint arXiv:2211.04686 (2022).
8. Yu, Da, et al. "Large scale private learning via low-rank reparametrization." International Conference on Machine Learning. PMLR, (2021).
9. Yue, Xiang, et al. "Synthetic text generation with differential privacy: A simple and practical recipe." arXiv preprint arXiv:2210.14348 (2022).
10. Qu, Chen, et al. "Natural language understanding with privacy-preserving bert." Proceedings of the 30th ACM International Conference on Information & Knowledge Management. (2021).
11. Feyisetan, Oluwaseyi, et al. "Privacy-and utility-preserving textual analysis via calibrated multivariate perturbations." Proceedings of the 13th international conference on web search and data mining. (2020).
12. Li, Yansong, Zhixing Tan, and Yang Liu. "Privacy-preserving prompt tuning for large language model services." arXiv preprint arXiv:2305.06212 (2023).
13. Mai, Peihua, et al. "Split-and-Denoise: Protect large language model inference with local differential privacy." arXiv preprint arXiv:2310.09130 (2023).
14. Shen, Xicong, et al. "A Split-and-Privatize Framework for Large Language Model Fine-Tuning." arXiv preprint arXiv:2312.15603 (2023).
15. Huang, Yangsibo, et al. "TextHide: Tackling data privacy in language understanding tasks." Findings of the Association for Computational Linguistics: EMNLP 2020. (2020).
16. Lee, Garam, et al. "Privacy-preserving text classification on BERT embeddings with homomorphic encryption." arXiv preprint arXiv:2210.02574 (2022).
17. Akimoto, Y., et al. "Privformer: Privacy-preserving Transformer with MPC." 2023 IEEE 8th European Symposium on Security and Privacy (EuroS&P). (2023).
18. Zhou, Xin, et al. "TextFusion: Privacy-Preserving Pre-trained Model Inference via Token Fusion." Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing. (2022).
19. Zhou, Xin, et al. "TextMixer: Mixing Multiple Inputs for Privacy-Preserving Inference." Findings of the Association for Computational Linguistics: EMNLP 2023. (2023).
20. Lukas, N., et al. "Analyzing Leakage of Personally Identifiable Information in Language Models." 2023 IEEE Symposium on Security and Privacy (SP). (2023).
21. Li, Haoran, et al. "P-bench: A multi-level privacy evaluation benchmark for language models." arXiv preprint arXiv:2311.04044 (2023).
22. Arnold, Stefan, Dilara Yesilbas, and Sven Weinzierl. "Guiding text-to-text privatization by syntax." arXiv preprint arXiv:2306.01471 (2023).
23. Yang, An, et al. "Qwen2.5 technical report." arXiv preprint arXiv:2505.09388 (2025).
24. Wang, Alex, et al. "GLUE: A multi-task benchmark and analysis platform for natural language understanding." arXiv preprint arXiv:1804.07461 (2018).
25. Wikimedia Foundation, “Wikimedia Downloads.” [Online]. Available: <https://dumps.wikimedia.org>
26. Reimers, Nils, and Iryna Gurevych. "Sentence-bert: Sentence embeddings using siamese bert-networks." arXiv preprint arXiv:1908.10084 (2019).
27. Douze, Matthijs, et al. "The faiss library." arXiv preprint arXiv:2401.08281 (2024).
28. D. Han, M. Han, and the Unsloth team, "Unsloth," 2023. [Online]. Available: <http://github.com/unslothai/unsloth>

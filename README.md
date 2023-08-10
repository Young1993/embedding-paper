# Embedding-paper

To track the latest paper for embedding (including text/text-code/text-image embeddings)

## Text embedding

|  paper   | 主要内容  | 论文来源 |
|  ----  | ----  | ---- |
| TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning  | 这篇论文作者是希望构建一个encoder-decoder结构来做无监督句子表示，他通过往输入句子中增加噪音，给decoder一个固定大小的sentence表示，然后来恢复原始输入，增强encoder的表征能力。一个去燥的做法。|EMNLP2021 findings |
| GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval| 这篇论文要解决的问题是通过无监督学习适应domain，他使用T5根据文章内容生成对应的query，对于生成的query，作者检索最相似的段落最为负样本| NAACL2022 |
| One Embedder, Any Task: Instruction-Finetuned Text Embeddings  | 作者不是研究某个特定的任务或领域，而是希望通过手写instruction，在330个任务上进行训练，通过instruction和多任务学习来统一embedding，使得模型在新的domain或任务时，具备很好的表示能力。 | ACL2023 |
| Text Embeddings by Weakly-Supervised Contrastive Pre-training|  这篇论文作者提出一种在无监督数据集上训练就能达到很好效果的模型E5，主要贡献就是构建了CCPairs数据集，挑选出了优质数据，batch size也比较大，在BEIR上超过了BM25，在 MTEB实现了SOTA | Arxiv2022 |
| PROMPTAGATOR : FEW-SHOT DENSE RETRIEVAL FROM 8 EXAMPLES| 信息检索的做法主要是利用监督数据，但是监督数据很稀缺，作者基于few-shot的范例来引导大模型来作为task-specific query generator来生成查询问题，作者设计了一个consistency的挑选，也就是通过生成的query检索出来的top-K的文章中一定要有document来挑选query-document对，然后利用高质量的内容训练retriever。作者用T5-base作为retriever效果会好于相同参数的模型。 | ICLR2023 |
|RetroMAE: Pre-Training Retrieval-oriented Language Models Via Masked Auto-Encoder|这篇论文的做法和bottleneck的做法很像，都是有一个encoder，加一个shallow的decoder，然后通过加mask等再充分训练encoder的embedding，从而提升效果|EMNLP2022|
### Document embedding
|  paper   | 主要内容  | 论文来源 |
|  ----  | ----  | ---- |
|CODER: An efficient framework for improving retrieval through COntextual Document Embedding Reranking| 对比学习在训练dense retrieval模型时用的很多，一般都是采用in-batch negative负样本的做法，作者做了3点改进，首先通过一个对文档的打分方法（基于预训练模型，计算一个query对一些document的联合相关性得分），从而可以检索到负样本而不是随机的负样本，这样采样质量更高；作者还采用了一个完全的list-wise loss来训练，取得了比较好的效果。（有点没get到这篇优势，可能最大的就是针对query去筛选样本，然后可以和现有的模型结合，从而得到效果提升）  | EMNLP2022 |
|Improving Document Representations by Generating Pseudo Query Embeddings for Dense Retrieval|这篇论文的做法很有意思，作者分析了做dense retrieval的发展历史,比如很多工作先使用bm25做retrieval，然后使用dense embedding 做reranker；作者分析了cross-encoder、bi-encoder和poly-encoder；他主要做法是对于文档进行encoder后，取document的token embeddings，然后对这些embedding进行聚类，每次的聚类中心当作一个伪query embedding，从而对文档生成了多个伪embedding；在推理时直接算query和伪embedding的相似度，实验证明这起到了很好的效果。 |ACL2021|
### Benchmark
|  paper   | 主要内容  | 论文来源 |
|  ----  | ----  | ---- |
| BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models  | 主要是构建一个包含多样数据的信息检索的benchmark，包含了词汇、稀疏、密集、后期交互和重新排序等场景，总共包含18个数据集。 | Nips2021 |
### Loss 创新

|  paper   | 主要内容  | 论文来源 |
|  ----  | ----  | ---- |
| MultipleNegativesRanking | 对于检索到的文章，匹配的文章被认为是相关的，其他的文章被任务为是不相关的，有一个问题就是，其他的文章和query其实有一些相关性。  |  |
| MarginMSE|使用一个cross-encoder对query和passage做一个soft-label，他计算从负样本到正样本之间的距离 和 原始cross-encoder中正样本、负样本的值接近 ||
|MultipleNegativesRankingLoss| 这个loss是当我们只有query和response，没有负样本的时候使用。对于sentence pairs (a_1, p_1), (a_2, p_2)…, (a_n, p_n)， (a_i, p_i) 是正样本对 而 (a_i, p_j) for i!=j 是负样本对.||
|CosineSimilarityLoss| 句子A和句子B直接比较他们的余弦相似度，但这里要求句子A和B之间有gold 的相似度||
## Text-code embedding

|  paper   | 主要内容  | 论文来源 |
|  ----  | ----  | ---- |
|  Text and Code Embeddings by Contrastive Pre-Training | OpenAI提出基于无监督的对比学习来构造高质量的向量表示，在text和code的向量化任务上取得了比较好的效果，这篇论文值得关注的地方就是用的batch size比较大，还有就是他指出句子相似度的任务对分类或检索来说是有矛盾的地方，因为有一些分类句子是相似的但对于检索来说没有用；作者降低了句子相似度任务的loss权重。 | Arxiv2022 |

## Text-image embedding
|  paper   | 主要内容  | 论文来源 |
|  ----  | ----  | ---- |
|  Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese | 本文主要的做法就是复现CLIP在中文文本和图像之间对齐的，首先用CLIP的image encode来初始化image encoder，用Chinese RoBERTa来初始化文本encoder，第一阶段冻住image encoder来tune 文本的encoder；第二阶段，一起tune直到收敛。 | Arxiv2022 |

# Retrieval-based Language Models and Applications
|  paper   | 主要内容  | 论文来源 |
|  ----  | ----  | ---- |
|  REALM: Retrieval-Augmented Language Model Pre-Training  | 语言模型因为在预训练阶段捕获了大量的世界知识，因此推理效果很强大；但这要求很大的模型参数来存储这些知识，这篇论文作者的思路是基于retriever+小模型的方式来提高QA任务的能力，具体做法就是在预测之前，模型从维基百科检索相关文档，然后把相关文档和query一起来帮助推理，实现了比较好的效果。| Arxiv2020 |

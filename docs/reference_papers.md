# Reference Papers

## Summarization & SFT (Core)
- T5 (Raffel et al., 2019): "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
  - arXiv: https://arxiv.org/abs/1910.10683
  - Why: canonical text-to-text framing; foundation for our T5/FLAN runs.
- FLAN / instruction tuning (Chung et al., 2022): "Scaling Instruction-Finetuned Language Models"
  - arXiv: https://arxiv.org/abs/2210.11416
  - Why: explains instruction-tuning data mixtures and generalization.
- BART (Lewis et al., 2019): "Denoising Sequence-to-Sequence Pre-training for Natural Language Generation"
  - arXiv: https://arxiv.org/abs/1910.13461
  - Why: major seq2seq pretraining baseline and summarization reference point.

## Evaluation (Automatic Metrics)
- ROUGE (Lin, 2004): "ROUGE: A Package for Automatic Evaluation of Summaries"
  - ACL Anthology: https://aclanthology.org/W04-1013/
  - Why: classic n-gram overlap for summarization.
- BLEU (Papineni et al., 2002): "Bleu: a Method for Automatic Evaluation of Machine Translation"
  - ACL Anthology: https://aclanthology.org/P02-1040/
  - Why: classic n-gram overlap for generation; still common in SFT baselines.
- BERTScore (Zhang et al., 2019): "BERTScore: Evaluating Text Generation with BERT"
  - arXiv: https://arxiv.org/abs/1904.09675
  - Why: embedding-based semantic similarity.
- BLEURT (Sellam et al., 2020): "Learning to Evaluate Translation Beyond English: BLEURT Submissions to the WMT Metrics 2020 Shared Task"
  - arXiv: https://arxiv.org/abs/2010.04297
  - Why: learned metric; stronger correlation with human judgments.

## Faithfulness / Factual Consistency
- FactCC (Kryscinski et al., 2019): "Evaluating the Factual Consistency of Abstractive Text Summarization"
  - arXiv: https://arxiv.org/abs/1910.12840
  - Why: weakly supervised factuality classifier for summaries.
- QAGS (Wang et al., 2020): "Asking and Answering Questions to Evaluate the Factual Consistency of Summaries"
  - arXiv: https://arxiv.org/abs/2004.04228
  - Why: QA-based faithfulness metric; more semantic than n-gram overlap.

## NLI / Entailment (Used for Faithfulness Judging)
- SNLI (Bowman et al., 2015): "A large annotated corpus for learning natural language inference"
  - ACL Anthology: https://aclanthology.org/D15-1075/
  - Why: widely used for entailment models.
- MultiNLI (Williams et al., 2018): "A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference"
  - ACL Anthology: https://aclanthology.org/N18-1101/
  - Why: multi-genre entailment data; common for robust NLI models.

## Optional: Local PDFs
- Local copies are stored in `docs/papers/`:
  - `docs/papers/t5_1910.10683.pdf`
  - `docs/papers/flan_t5_2210.11416.pdf`
  - `docs/papers/bart_1910.13461.pdf`
  - `docs/papers/rouge_w04-1013.pdf`
  - `docs/papers/bleu_p02-1040.pdf`
  - `docs/papers/bertscore_1904.09675.pdf`
  - `docs/papers/bleurt_2010.04297.pdf`
  - `docs/papers/factcc_1910.12840.pdf`
  - `docs/papers/qags_2004.04228.pdf`
  - `docs/papers/snli_d15-1075.pdf`
  - `docs/papers/multinli_n18-1101.pdf`
- These PDFs are commit-ready; consider Git LFS if you want a very lightweight repo.

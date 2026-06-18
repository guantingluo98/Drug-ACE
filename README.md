# Drug-ACE
This is the official repository for the paper **["Applicability Condition Extraction for Therapeutic Drug-Disease Relations"](https://arxiv.org/abs/2606.14031)**, accepted by **ACL 2026 Findings**.

# ⚙️Requirements
We recommend using `Python 3.10.16`, You can set up the environment by following these steps:

```bash
pip install torch==2.6.0+cu126 --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)

pip install transformers==4.51.3

pip install --no-deps trl==0.15.2
```

# 📊Dataset
The Drug-ACE dataset has been publicly released on Hugging Face

https://huggingface.co/datasets/B1tta/Drug-ACE

# Citation

If you find our paper, code, or dataset helpful, please consider citing our work:

```bibtex
@misc{luo2026applicabilityconditionextractiontherapeutic,
      title={Applicability Condition Extraction for Therapeutic Drug-Disease Relations}, 
      author={Guanting Luo and Noriki Nishida and Yuji Matsumoto and Yuki Arase},
      year={2026},
      eprint={2606.14031},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={[https://arxiv.org/abs/2606.14031](https://arxiv.org/abs/2606.14031)}, 
}

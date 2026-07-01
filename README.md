# Drug-ACE
This is the official repository for the paper **["Applicability Condition Extraction for Therapeutic Drug-Disease Relations"](https://aclanthology.org/2026.findings-acl.154/)**, accepted by **ACL 2026 Findings**.

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
@inproceedings{luo-etal-2026-applicability,
    title = "Applicability Condition Extraction for Therapeutic Drug-Disease Relations",
    author = "Luo, Guanting  and
      Nishida, Noriki  and
      Matsumoto, Yuji  and
      Arase, Yuki",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.154/",
    pages = "3135--3148",
    ISBN = "979-8-89176-395-1",
    abstract = "Identifying conditions that a certain drug takes therapeutic effect on a target disease is crucial for clinical decision-making support. However, most existing biomedical information extraction methods have focused on identifying only relations between drugs and diseases, while largely overlooking the context-specific conditions where such relations can apply. To address this problem, we introduce the task of applicability condition extraction for therapeutic drug{--}disease relations from biomedical research literature. We create the first dataset that has manually annotated triples of drugs, diseases, and applicability conditions on biomedical paper abstracts with 1,119 drug-disease pairs. Using this dataset, we systematically evaluate the performance of a range of existing methods. In addition, we propose a new method that enhances LoRA to consider relations between drugs and diseases. Our method consistently outperforms strong baselines across different evaluation settings."
}

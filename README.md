# DistilBERT--SQuAD-v1-Notebook



<img src="https://huggingface.co/front/assets/huggingface_logo.svg">



This Notebook contains set of instructions how to train [DistilBert](https://huggingface.co/transformers/v2.10.0/model_doc/distilbert.html) from Huggingface in Google Colab.
Training is done on the [SQuAD](https://huggingface.co/datasets/squad) dataset. The model can be accessed via [HuggingFace](https://huggingface.co/abhilash1910/distilbert-squadv1):


```python

from transformers import AutoModelForQuestionAnswering,AutoTokenizer,pipeline
model=AutoModelForQuestionAnswering.from_pretrained('abhilash1910/distilbert-squadv1')
tokenizer=AutoTokenizer.from_pretrained('abhilash1910/distilbert-squadv1')
nlp_QA=pipeline('question-answering',model=model,tokenizer=tokenizer)
QA_inp={
    'question': 'What is the fund price of Huggingface in NYSE?',
    'context': 'Huggingface Co. has a total fund price of $19.6 million dollars'
}
result=nlp_QA(QA_inp)
result
```

The result is:

```bash

{'score': 0.38547369837760925,
 'start': 42,
 'end': 55,
 'answer': '$19.6 million'}
 ```


## distilBERT

DistilBERT is a lighter version of BERT which uses 40% less size from BERT but retains 97% of its performance. The original paper can be found [here](https://arxiv.org/abs/1910.01108).

Tips:

- DistilBERT doesn’t have token_type_ids, you don’t need to indicate which token belongs to which segment. Just separate your segments with the separation token tokenizer.sep_token (or [SEP]).

- DistilBERT doesn’t have options to select the input positions (position_ids input)


The architecture involves training a "student BERT" with reduced parameters from the pretrained "teacher BERT" (weight transfer):

<img src="https://miro.medium.com/max/1600/1*r8kVneErBpqN7KjrxHe_Sw.png">

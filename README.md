# Transformers Pre-Training: Masked Language Modeling

Masked Language Modeling is a pre-training technique to teach transformers the semantics of a language by essentially asking it to *fill in the blanks*.

Then the pre-trained model can be finetuned for downstream tasks such as classification, generation, etc.

## Kaggle Notebook: [Masked Language Modeling from Scratch](https://www.kaggle.com/shreydan/masked-language-modeling-from-scratch)

```py
>>> predict_mask('hello! how are you?')
masked: you predicted: you
ACTUAL: hello! how are you?
MASKED: hello! how are [MASK]?                                        
 MODEL: hello! how are you?

```

### Objectives:

- Implementing Encoder-Only Transformer model ✅️
- Preparing the dataset from scratch ✅️
- Training BERT-like tokenizer from scratch ✅️
- Training from scratch with MLM objective ✅️
- Trained on Wikipedia dataset from scratch ✅️

### Preparing Dataset

- in Masked Language Modeling, the loss calculation is similar to that of causalLMs
- the inputs and the labels are identical in terms of position
- in the inputs:
    - 15% of the tokens are masked / replaced randomly by the [MASK] token
    - this 15% doesn't include the pad tokens
- in the labels:
    - the ground-truth tokens which were masked in the inputs are present in the labels
    - all other tokens are ignored (set to -100) by the default behaviour of nn.CrossEntropyLoss

#### This is how a sample looks

- 1 is the [PAD] token
- 2 is the [MASK] token
- model max length = 128

```py
tensor([5680,   10,  313,    2,    2, 4541,   14, 5393, 5404,   70,   11,  153,
          40, 2319,    2, 7560,   14, 1681, 3534,  148, 1649,   16,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1])
tensor([-100, -100, -100,  939, 1058, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, 1246, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100])
```


### Encoder-Only Transformer Model

- straightforward simple implementation.
- nn.LayerNorm replaced with RMSNorm which is preferred to by many. Code from [here](https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py).
- It looks like BERT but it is not BERT. BERT is more complicated than this.
- Only implementing the MLM part of BERT so no need of [CLS] and [SEP] tokens
- Learned positional embeddings instead of sinusoidal in BERT.
- We can have a mask for the encoder self-attention as well by masking out the pad tokens so attention layers ignore the extra stuff.
- For inference currently only supports batch size of 1.
- After the encoder outputs pass through the dim->vocab Linear layer, the logits at the position where the token was masked are softmaxed and then with argmax the token that's supposed to be there is predicted.

```
out: 1 x 128 x 256
if the input sequence for inference was masked at position 4, we extract 1 x 256 at index 4:
preds: out[:,4,:]
softmax -> argmax
preds: predicted token
```

## Examples
> while the predicted word may not be exact, it conveys the meaning of the sentence pretty well as it understands the context of the entire sentence.

```
masked: feed predicted: feed
ACTUAL: The larvae are black and flattened and feed on snails as well.
MASKED: the larvae are black and flattened and [MASK] on snails as well.                                
MODEL: the larvae are black and flattened and feed on snails as well.

---

masked: facility predicted: school
ACTUAL: Throughout the year, the facility hosts a variety of educational programs.
MASKED: throughout the year, the [MASK] hosts a variety of educational programs.MODEL: throughout the year, the school hosts a variety of educational programs.

---

masked: provide predicted: create
ACTUAL: IRIS has partnered with other NGOs to provide funding for services, such as with The Vision Charity in Sri Lanka.
MASKED: iris has partnered with other ngos to [MASK] funding for services, such as with the vision charity in sri lanka.                       
MODEL: iris has partnered with other ngos to create funding for services, such as with the vision charity in sri lanka.

```

you can check the notebook for more output examples.
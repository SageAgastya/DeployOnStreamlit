from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re, numpy
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Model:
    def __init__(self, ckpt):
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)

    def __call__(self, data, hypo, labels):
        premise = data
        hypothesis = hypo             # hypo contains a <mask> token and doesn't contain class label.
        
        score_list = []
        for label in labels:
            hypothesis = re.sub("<mask>", label, hypothesis)
            # run through model pre-trained on MNL
            x = self.tokenizer.encode(premise, hypothesis, return_tensors='pt',
                                 truncation_strategy='only_first')
            logits = self.nli_model(x.to(device))[0]

            # we throw away "neutral" (dim 1) and take the probability of
            # "entailment" (2) as the probability of the label being true 
            entail_contradiction_logits = logits[:,[0,2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = probs[:,1].tolist()
#             print(prob_label_is_true)
            score_list.append(prob_label_is_true[0])
            hypothesis = re.sub(label, "<mask>", hypothesis)
        
        max_idx = numpy.array(score_list).argmax()
        return (labels[max_idx], score_list[max_idx])
    
# model = Model("roberta-base")
# print(model("I am happy.", "My sentiment is <mask>", ["positive", "negative"]))
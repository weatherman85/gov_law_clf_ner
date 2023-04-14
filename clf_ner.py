from transformers import BertPreTrainedModel
from typing import List
import torch
from torch import nn
import numpy as np
from transformers import AutoTokenizer,BertModel

class ClassifierNER(BertPreTrainedModel):
    def __init__(self,config):
        super(ClassifierNER,self).__init__(config)                  
        self.bert = BertModel(config, add_pooling_layer=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()
        # set classifier layer
        self.clf_labels= config.clf_labels
        self.clf_classes = len(self.clf_labels)
        self.clf_linear = nn.Linear(config.hidden_size,self.clf_classes)
        #set ner layer
        self.ner_labels = config.ner_labels
        self.ner_classes = len(self.ner_labels)
        self.ner_linear = nn.Linear(config.hidden_size,self.ner_classes)
        self.ner_lstm = nn.LSTM(config.hidden_size,config.hidden_size//2,dropout=config.hidden_dropout_prob,batch_first=True,bidirectional=True)       
    def forward(self,input_ids,token_type_ids,attention_mask,clf_labels=None,ner_labels=None,**kwargs):
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,**kwargs)
        clf_output = outputs[1]
        clf_output = self.dropout(clf_output)
        clf_logits = self.clf_linear(clf_output)
        clf_loss = 0
        if clf_labels is not None:
          clf_labels_tensor = torch.tensor(clf_labels, dtype=torch.long)
          clf_loss = self.loss_fct(clf_logits.view(-1, self.clf_classes), clf_labels_tensor.view(-1))
        ner_output = outputs[0]
        ner_output = self.dropout(ner_output)
        lstm_output,hc = self.ner_lstm(ner_output)
        ner_logits = self.ner_linear(lstm_output)        
        ner_loss = 0
        if ner_labels is not None:
            ner_loss = self.loss_fct(ner_logits.view(-1,self.ner_classes),ner_labels.view(-1)) 
        if clf_labels is not None or ner_labels is not None:
            loss = clf_loss + ner_loss
            return loss, clf_logits, ner_logits
        else:
            return clf_logits,ner_logits
    def predict(self,text):
        with torch.no_grad():
            tokenized = self.tokenizer.encode_plus(text,truncation=True,max_length=512,return_tensors="pt",return_offsets_mapping=True)
            clf_prediction,ner_prediction = self(tokenized['input_ids'],tokenized['token_type_ids'],tokenized['attention_mask'])
            clf_prediction = self.clf_labels[str(torch.argmax(clf_prediction,dim=-1).item())]
            ner_prediction = self.align_predictions(text,ner_prediction,tokenized['offset_mapping'])
        return {"classification":clf_prediction,"entities":ner_prediction}    
    def align_predictions(self,text,predictions,offsets):
        results = []
        predictions = torch.argmax(predictions,dim=-1)[0].tolist()
        offsets = offsets[0].tolist()
        idx = 0
        while idx < len(predictions):
            pred = predictions[idx]
            label = self.ner_labels[str(pred)]
            if label != "O":
                # Remove the B- or I-
                label = label[2:]
                start, end = offsets[idx]
                # Grab all the tokens labeled with I-label
                idx += 1
                while (
                    idx < len(predictions)
                    and self.ner_labels[str(predictions[idx])] == f"I-{label}"
                ):
                    _, end = offsets[idx]                    
                    idx += 1
                
                # The score is the mean of all the scores of the tokens in that grouped entity
                word = text[start:end]
                results.append(
                    {
                        "label": label,
                        "entity": word,
                        "start": start,
                        "end": end,
                    }
                )
            idx += 1
        return results
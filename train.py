import torch
import numpy as np
from datasets import load_dataset, load_metric
from transformers import Wav2VecProcessor, Wav2Vec2ForCTC, Wav2VecforCTCTokenizer,Wav2Vec2FeatureExtractor,TrainingArguments, Trainer
import re 
import json
import dataclasses
from typing import Dict
import torchaudio


torch.manual_seed(42)
np.random_seed(42)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

output_dir="./wav2vec"

librispeechtrain= load_dataset("librispeech_asr", clean100, split="train")
librispeechval=load_dataset("librispeech_asr", clean100,  split="val")
librispeechtest=load_dataset("librispeech_asr", clean100, split="test")

def extractchar(batch):
    all_text=" ".join(batch["text"])
    vocab=list(set(all_text))

vocab_train=librispeechtest.map(extractchar, batched=True,
                                batch_size=1000,
                                keep_in_memory=True,
                                remove_columns=librispeechtrain.column_names)

vocab_val = librispeechval.map(
   extractchar,
   batched=True,
   batch_size=1000,
   keep_in_memory=True,
   remove_columns=librispeechval.column_names
)

vocablist=list(set(vocab_train["vocab"][0])) | set(vocab_val["vocab"][0])
vocabdict={v: k for k , v in enumerate(sorted(vocablist))
           }
with open(os.path.join(output_dir, vocab_dir, vocab.json)):
    json.dump(vocabdict, vocab_file)

#tokenizer
tokenizer=Wav2VecforCTCTokenizer(
    os.path.join(output_dir, "vocab.json"),
    pad_token="<pad>",
    word_delimiter_token="|",
    unk_token="<unk>"
    )
#featureExtract
feature_extractor= Wav2VecFeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False
)

#processor
processor=Wav2VecProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor.save_pretrained(output_dir)

def preparedataset(examples):
    audio=examples["audio"]
    inputs=processor(
        [sample["array"] for sample in audio],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    #target text
    test_examples=examples["text"]
    targets=processor.tokenizer(test_examples,padding= True)

    return{
        "input values": inputs.input_values.squeeze(),
        "attention_mask": inputs.attention_mask.squeeze,
        "labels": targets["input_ids"],
    }

train_dataset=librispeechtrain.map(
    preparedataset,
    batch_size=8,
    batched=True,
    num_proc=4
)
val_dataset=librispeechval.map(
    preparedataset,
    batch_size=8,
    batched=True,
    num_proc=4
)
@dataclass
class Datacollator:
    processor: Wav2VecProcessor
    padding: Union[bool, str]= True

    def __call__(self, features: List[dict[str, Union[list[int],torch .tensor]]]) -> Dict[str, torch.Tensor]:
        input_features=[{"input_values": feature["input_values"]} for feature in features]
        label_features=[{"input_ids": feature['labels']} for feature in features]

        batch= self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",

        )
        with self.processor.as_targer_processor():
            labels=labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1),-100)
            batch["labels"]=labels
            return batch

Data_collator=Datacollator(processor=processor, padding=True)
model = Wav2Vec2ForCTC.from_pretrained("base",
                                       ctc_loss_reduction="mean",
                                       pad_token_id=processor.tokenizer.pad_token_id, vocab_size=len(processor.tokenizer),
                                       )

model.freeze_feature_encoder()
wer_metric=load_metric("wer")

def computemetrics(pred):
    pred_logits=pred.predictions
    pred_ids=np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100]=processor.tokenizer.pad_token_id

    pred_str=processor.batch_decode(pred_ids, group_tokens=False)

    wer=wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer":wer}

training_args=TrainingArguments(
    output_dir=output_dir,
    group_by_length=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    num_train_epochs=30,
    fp16=True,
    learning_rate=3e-4,
    report_to="tensorboard",
    gradient_checkpoint=True,

)

trainer= Trainer(
    model=model,
    Data_collator=Data_collator,
    args=training_args,
    compute_metrics=computemetrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.feature_extractor,

)

#training
trainer.train()

model.save_pretrained(os.path.join(output_dir, "final_model"))

#evaluation on test data
test_dataset=librispeechtest.map(
    preparedataset,
    batch_size=8,
    batched=True,
    num_proc=4,

)


result=trainer.evaluate(test_dataset)

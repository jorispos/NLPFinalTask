import json

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments

model_dir = './my_roberta_model'

model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

squad = load_dataset('squad_v2')

# This function was partially based off of: https://stackoverflow.com/questions/77484646/how-should-i-preprocess-this-dataset-for-performing-a-question-answering-task
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        if not answer["answer_start"]:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


tokenized_squad = squad.map(preprocess_function, batched=True)


training_args = TrainingArguments(
    output_dir='./',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad['train'],
    eval_dataset=tokenized_squad['validation']
)

output = trainer.predict(tokenized_squad['validation'])

print("output: ", output)

predictionsNP = output.predictions
np.save('predictions', predictionsNP)

print("Predictions written to predictions.npy")

predictionsJSON = []
for i, (start_logits, end_logits) in enumerate(zip(output.predictions[0], output.predictions[1])):
    all_tokens = tokenizer.convert_ids_to_tokens(tokenized_squad['validation'][i]['input_ids'])
    answer_start = torch.argmax(torch.tensor(start_logits)).item()
    answer_end = torch.argmax(torch.tensor(end_logits)).item() + 1
    answer = tokenizer.convert_tokens_to_string(all_tokens[answer_start:answer_end])
    predictionsJSON.append(answer)

pred_dict = {}
for i, qa_id in enumerate(tokenized_squad['validation']['id']):
    pred_dict[qa_id] = predictionsJSON[i]

with open('predictions.json', 'w') as f:
    json.dump(pred_dict, f)

print("Predictions written to predictions.json")

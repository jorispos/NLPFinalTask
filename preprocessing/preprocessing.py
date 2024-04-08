from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_name = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

squad = load_dataset('squad_v2')

train_size = int(0.5 * len(squad['train']))
squad['train'] = squad['train'].select(range(train_size))
val_size = int(0.5 * len(squad['validation']))
squad['validation'] = squad['validation'].select(range(val_size))

# This function was partially based off this source: https://stackoverflow.com/questions/77484646/how-should-i-preprocess-this-dataset-for-performing-a-question-answering-task
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

model = AutoModelForQuestionAnswering.from_pretrained(model_name)

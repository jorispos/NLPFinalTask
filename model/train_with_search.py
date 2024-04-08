from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import optuna

model_name = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
squad = load_dataset('squad_v2')


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


def hyperparam_search(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    num_train_epochs = trial.suggest_int('num_train_epochs', 1, 5)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32])

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        learning_rate=learning_rate,
        logging_steps=500,
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_squad['train'],
        eval_dataset=tokenized_squad['validation']
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result["eval_loss"]


study = optuna.create_study(direction="minimize")
study.optimize(hyperparam_search, n_trials=10)

best_trial = study.best_trial
print("Best trial final validation loss:", {best_trial.value})
print("Best trial hyperparameters:", {best_trial.params})

model.save_pretrained('./my_best_roberta_model')
tokenizer.save_pretrained('./my_best_roberta_model')

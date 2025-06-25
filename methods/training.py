from transformers import TrainingArguments, Trainer

def get_training_args(output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=2,
        num_train_epochs=5,
        per_device_train_batch_size=15,
        gradient_accumulation_steps=2,
        learning_rate=5e-4,
        logging_steps=100,
        fp16=True,
        seed=42,
        push_to_hub=False,
        report_to=[]
    )
    return training_args

def get_trainer(model,training_args,tokenized_dataset,tokenizer,data_collator,StepPrinterCallback):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[StepPrinterCallback()]
    )
    return trainer

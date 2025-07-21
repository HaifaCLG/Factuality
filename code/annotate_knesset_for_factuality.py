import json
import os

import numpy as np
from datasets import Dataset

from check_worthiness_classification_training import load_saved_model_and_tokenizer, compute_metrics, tokenize_function

MODEL_NAME = "GiliGold/Knesset-DictaBERT"
MODEL_SAVE_PATH = "./fine_tuned_knesset-dicta_bert"
from transformers import Trainer

knesset_protocols_path = "data\\processed_knesset\\NER_protocols\\plenary"
factuality_protocols_path = "data\\processed_knesset\\factuality_protocols\\plenary"

if __name__ == '__main__':
    model, tokenizer = load_saved_model_and_tokenizer(MODEL_SAVE_PATH)
    trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    for protocol in os.listdir(knesset_protocols_path):

        protocol_sentences_texts = []
        protocol_path = os.path.join(knesset_protocols_path, protocol)
        factuality_protocol_path =os.path.join(factuality_protocols_path, f'factuality_{protocol}')
        if os.path.exists(factuality_protocol_path):
            print(f'skip {protocol}')
            continue

        print(f'starting to predict check-worthiness for {protocol}')
        try:
            with open(protocol_path, encoding="utf-8") as file:
                protocol_json = json.load(file)
        except Exception as e:
            print(f'couldnt parse json: {protocol_path}')
            continue

        for sent in protocol_json["protocol_sentences"]:
            sent_text = sent["sentence_text"]
            protocol_sentences_texts.append(sent_text)
            # Prepare a Hugging Face dataset from your sentences
        protocol_dataset = Dataset.from_dict({"text": protocol_sentences_texts})
        protocol_dataset = protocol_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        protocol_dataset = protocol_dataset.remove_columns(["text"])
        try:
            protocol_predictions = trainer.predict(protocol_dataset)
        except Exception as e:
            print(f'couldnt predict factuality for {protocol}. error was: {e}')
            print(f'skipping file')
            continue
        protocol_predicted_labels_ids = np.argmax(protocol_predictions.predictions, axis=1)
        id2label = model.config.id2label
        protocol_predicted_labels = [id2label[pred] for pred in protocol_predicted_labels_ids]

        for sent, label in zip(protocol_json["protocol_sentences"], protocol_predicted_labels):
            if 'factuality_fields' in sent:
                sent['factuality_fields']={}
                sent['factuality_fields']['check_worthiness_score'] = label
        try:
            with open(factuality_protocol_path, 'w', encoding='utf-8') as file:
                json.dump(protocol_json, file, ensure_ascii=False)
        except Exception as e:
            print(f"couldnt dump json {protocol} error: {e}")
            continue
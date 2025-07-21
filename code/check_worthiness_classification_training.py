import json
import os
import shutil

from sklearn.metrics import cohen_kappa_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import evaluate

# Define global constants
# MODEL_NAME = 'dicta-il/dictabert'
MODEL_NAME = "GiliGold/Knesset-DictaBERT"
# MODEL_NAME = "dicta-il/alephbertgimmel-base"
MAX_LENGTH = 512
BATCH_SIZE = 8
NUM_EPOCHS = 3
# MODEL_SAVE_PATH = "./fine_tuned_dicta_bert"
MODEL_SAVE_PATH = "./fine_tuned_knesset-dicta_bert"
# MODEL_SAVE_PATH = "./fine_tuned_alephbertgimmel-base"

PROPOSITION_MODEL_SAVE_PATH = "./proposition_fine_tuned_knesset-dicta_bert"
WORTH_CHECKING_PROPOSITIONS_MODEL_SAVE_PATH = "./worth_checking_propositions_fine_tuned_knesset-dicta_bert"

test_sentences = ["רק לדבר אתו, שיגיד להם מה הוא רוצה.",
                 "אדוני היושב-ראש, כנסת נכבדה, דיברו כאן קודם על שתי קבוצות לכאורה: אחת של המשפטנים ומי ששייך למערכת המשפטית, למערכת אכיפת החוק או למערכת בתי-המשפט, והשנייה – פוליטיקאים, ועל איזה ניגוד מובנה בין שתי הקבוצות.",
                 "בדיוק הצעת חוק כזאת עלולה להגביר את המוטיבציה לבצע חטיפות בעתיד.",
                 "מה הוא ידע על ירושלים?",
                 "יש הרבה מאוד ספרות יהודית.",
                 "הוא הסכים להיות מעורב בעניין רק כדי שהם לא יתאבדו.",
                 "הערוץ השלישי התחיל בצליעה.",
                 "הדבר השני, לגבי רמת הדיון: פה אני פחות אופטימית.",
                 "החשש היה שהמנהל הכללי של משרד הבריאות \"יסונדל\".",
                 "שאלה שלישית – מה הם מנגנוני הבקרה הפרלמנטריים על מאסר כזה ועל אי-פרסום כזה?",
                 "בצנעה ובענווה ובשקט נפשי הצלחת להרגיע את הרוחות ולהביא את משרד המשפטים וכל פקידיו בחזרה למסלול.",
                 "שאלה רביעית – איך תתאפשר ביקורת ציבורית על מאסרים כאלה?",
                 "דיוניה סגורים.",
                 "מדוע הדבר הזה לא נעשה?",
                 "עכשיו לגופו של עניין.",
                 "כלומר, הם עברו את רוב הדרך לקראת הרישוי החוקי שלהם.",
                 "החובות משולמים.",
                 "גם בימים אלה אני מתמודד עם הקטע הזה.",
                 "5,000 שקל – מה כבר אפשר לעשות עם הדבר הזה?",
                 "זו הסתייגות אחת.",
                 "יש פה בעיה.",
                 "עובדה, יש 25,000 או 23,000 תכנונים שכבר אושרו.",
                 "אני חוזר ואומר: אדרבה, תפנה אותי לאזור מסוים, ספציפי.",
                 "אז כאן הגענו למסקנה שנראה איך המשק יעבוד סביב נושא שתי השכבות.",
                 "תודה.",
                 "תאמינו לי, זה שטויות.",
                 "זה משהו באמת, אני רוצה, אני לא מבין את זה בכלל.",
                 "כי לא מחוקקים סתם חוק.",
                 "עם כל הכבוד.",
                 "פשוט לא הוגש דו\"ח.",
                 "מה עושים?",
                 "לא פעם כשנעדרת מישיבות הסיעה היית במשלחות כאלה ואחרות של הסוכנות, בעבודה לעלייה לארץ ישראל.",
                 "כלומר, בנק ערבי הוא הבנק הרווחי ביותר בישראל.",
                 "ממש עשיתי בשבילכם עבודה כלכלית.",
                 "אני רוצה להגיב בקצרה.",
                 "אבי שמחון עשה עבודה בשם ראש הממשלה.",
                 "יצרת אותם בדמותך.",
                 "אנחנו נמצאים עכשיו בשלב האבל על הרוגי הרקטות מעזה, השם ייקום דמם.",
                 "בעיניי תמיד היית מין אינדיאנה ג'ונס ישראלי כזה, הרפתקן, אדם שאוהב את החיים, נטול פחד וסקרן לגלות מקומות חדשים.",
                 "אדוני היושב-ראש, חבריי השרים וחברי הכנסת, ברשותך, חבר הכנסת יבגני סובה, אני רק מברך אותך על נאומך הראשון.",
                 "האמת, הפעם ראשונה שפגשתי את היבה הייתה כשקיבלנו אותה לעבודה.",
                 "בסמכות הוועדה הזאת לטפל בסוגיה הזאת.",
                 "סוף אוקטובר רע לך?",
                 "בסוף דבריו כל אחד יוכל להתייחס.",
                 "יש פה איזושהי מן צרימה מאוד מאוד חזקה בין האופן בו אנחנו מנהלים את הדיונים האלה, לוקחים את הזמן בקופות ולוקחים את הזמן במשרד הבריאות.",
                 "מאות אם לא אלפי התראות כבר חולקו.",
                 "אני פעיל בשוק ההון משנת 1968, עוד מתקופת שירותי הצבאי ועד שהתמניתי לשר – כ-47 שנה.",
                 "אני מאחל בהצלחה גם לכל חברי הוועדה.",
                 "כל מי שמעסיק היום עובד זר, ולא משנה באיזה תחום, אם בתחום המסעדנות או במפעלים או בעסקים שדיברת עליהם, וגם החקלאים, אמור לשלם 200% לאותם אנשים שבעצם אין להם בכלל זכות בחירה.",
                 "אנחנו הריבון.",
                 "אני מאמינה, אדוני, שגם אתה זוכר.",
                 "היא פועלת ללא סמכות בהרבה מאוד תחומים, ללא פיקוח וללא בקרה.",
                 "על האדם הזה כחול לבן תולים את יהבם.",
                 "הם שיקרו ומשקרים לציבור.",
                 "אם יש משהו שדווקא הקורונה לימדה אותנו, שאפשר להתאים את עצמנו, גם לרוח הזמן, גם לאמצעים הטכנולוגיים וגם, שומו שמים, למצוא פתרונות יצירתיים.",
                 "לא נשגע את הציבור."]
test_labels = ['not worth checking',
'worth checking',
'not worth checking',
'not a factual proposition',
'worth checking',
'not worth checking',
'worth checking',
'not worth checking',
'worth checking',
'not a factual proposition',
'worth checking',
'not a factual proposition',
'not worth checking',
'not a factual proposition',
'not worth checking',
'worth checking',
'worth checking',
'not worth checking',
'not a factual proposition',
'not worth checking',
'not worth checking',
'worth checking',
'not a factual proposition',
'worth checking',
'not a factual proposition',
'not worth checking',
'not worth checking',
'not worth checking',
'not a factual proposition',
'worth checking',
'not a factual proposition',
'worth checking',
'worth checking',
'not worth checking',
'not worth checking',
'worth checking',
'not worth checking',
'worth checking',
'not worth checking',
'not a factual proposition',
'worth checking',
'worth checking',
'not a factual proposition',
'not worth checking',
'worth checking',
'worth checking',
'worth checking',
'not a factual proposition',
'worth checking',
'worth checking',
'not worth checking',
'worth checking',
'worth checking',
'worth checking',
'worth checking',
'not worth checking']
def load_tokenizer(model_name):
    """Loads and returns the tokenizer for the model."""
    return AutoTokenizer.from_pretrained(model_name)
def tokenize_function(examples, tokenizer):
    """Tokenizes input text."""
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)


def prepare_dataset(sentences, labels, tokenizer, test_size=0.2, save_test_path=None):
    """Prepares the dataset for fine-tuning."""

    # Map labels to numerical values
    labels_list = sorted(list(set(labels)))
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}
    numerical_labels = [label2id[label] for label in labels]

    # Split into training and validation sets
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(sentences, numerical_labels,
                                                                        test_size=test_size, random_state=42)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42)

    # Convert to Hugging Face dataset format
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    if save_test_path is not None:
        if os.path.exists(save_test_path):
            shutil.rmtree(save_test_path)  # Ensure clean save
        test_dataset.save_to_disk(save_test_path)


    # Tokenize datasets
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Remove text column after tokenization
    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    }), label2id, id2label

def load_model(model_name, num_labels, label2id, id2label):
    """Loads the pre-trained model and moves it to GPU if available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    ).to(device)

    return model

def compute_metrics(eval_pred):
    """Computes accuracy for model evaluation."""
    accuracy_metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


def train_model(sentences, labels, model_name=MODEL_NAME, model_save_path= MODEL_SAVE_PATH):
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU mode")

    """Main function to fine-tune the model."""

    # Load tokenizer
    tokenizer = load_tokenizer(model_name)

    # Prepare dataset
    dataset, label2id, id2label = prepare_dataset(sentences, labels, tokenizer)

    # Load model
    model = load_model(model_name, num_labels=len(label2id), label2id=label2id, id2label=id2label)
    if model_save_path == WORTH_CHECKING_PROPOSITIONS_MODEL_SAVE_PATH:
        num_epochs = 4
    else:
        num_epochs = NUM_EPOCHS
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()
    # Evaluate and print accuracy
    eval_results = trainer.evaluate()
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")

    # Save final model
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print("Training completed and model saved successfully!")
def prepare_sentences_and_checkworthiness_labels_dataset(factuality_jsons_path):
    sentences = []
    labels = []
    annotators_dirs = os.listdir(factuality_jsons_path)
    for annotator_dir in annotators_dirs:
        factuality_json_files = os.listdir(os.path.join(factuality_jsons_path,annotator_dir))
        for json_file in factuality_json_files:
            factuality_file_path = os.path.join(factuality_jsons_path, annotator_dir, json_file)
            try:
                with open(factuality_file_path, encoding="utf-8") as file:
                    sent_entity = json.load(file)
            except Exception as e:
                print(f'in {annotator_dir}-{json_file} error: {e}')
                continue
            sent_text = sent_entity.get("sent_text", "")
            label = "not worth checking"#default label
            check_worthiness_profiles = sent_entity.get("check_worthiness",[])
            for check_worthiness_profile in check_worthiness_profiles:
                check_worthiness_score = check_worthiness_profile.get("check_worthiness_score", None)
                if check_worthiness_score == "worth checking":
                    label = "worth checking"
                    break
                elif check_worthiness_score == "not a factual proposition":
                    label = "not a factual proposition"
            if sent_text and check_worthiness_profiles:
                sentences.append(sent_text)
                labels.append(label)
    return sentences, labels

def load_saved_model_and_tokenizer(saved_path=MODEL_SAVE_PATH):
    tokenizer = AutoTokenizer.from_pretrained(saved_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(saved_path).to(device)
    return model, tokenizer
def evaluate_model(sentences, labels, saved_model_path=MODEL_SAVE_PATH):
    model, tokenizer = load_saved_model_and_tokenizer(saved_model_path)
    # Prepare the validation dataset
    dataset_dict, _, _ = prepare_dataset(sentences, labels, tokenizer, save_test_path="factuality_10_perc_test_set")
    validation_dataset = dataset_dict["validation"]
    test_dataset = dataset_dict["test"]

    # Setup a Trainer for evaluation
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Evaluate and print validation accuracy
    eval_results = trainer.evaluate(eval_dataset=validation_dataset)
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")

    test_results = trainer.evaluate(eval_dataset=test_dataset)
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)

    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")

    kappa_model_labels = cohen_kappa_score(predicted_labels, test_dataset["label"])
    print(f'Kappa between predicted labels and true labels: {kappa_model_labels}')




def save_sentences_labels(sentences, labels, path="sentences_labels.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"sentences": sentences, "labels": labels}, f, ensure_ascii=False, indent=4)

def load_sentences_labels(path="sentences_checkworthiness_labels.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["sentences"], data["labels"]

def evaluate_model_on_sentences(sentences, model_saved_path=MODEL_SAVE_PATH):
    from datasets import Dataset
    import numpy as np


    # Load the saved model and tokenizer
    model, tokenizer = load_saved_model_and_tokenizer(saved_path=model_saved_path)

    # Prepare a Hugging Face dataset from your sentences
    dataset = Dataset.from_dict({"text": sentences})
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    dataset = dataset.remove_columns(["text"])

    # Set up a Trainer for prediction
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
    )

    # Get predictions from the model
    predictions = trainer.predict(dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)

    # Retrieve the id2label mapping (saved in model config)
    id2label = model.config.id2label

    # Convert numerical predictions to string labels
    predicted_labels = [id2label[pred] for pred in pred_labels]

    print("Predicted labels:", predicted_labels)
    right_prediction_counter = 0
    for predicted_label, real_label in zip(predicted_labels, test_labels):
        if predicted_label == real_label:
            right_prediction_counter +=1

    accuracy = right_prediction_counter/len(test_labels)
    print(f"Test Accuracy by majority vote label: {accuracy:.4f}")


def split_dataset_to_factual_propositions_and_not_factual_propositions_labels(sentences, labels):
    new_labels = []
    for sent, label in zip(sentences, labels):
        if label == 'not a factual proposition':
            new_labels.append(label)
        else:
            new_labels.append("factual proposition")
    return sentences, new_labels

def two_phase_models_evaluation(test_sentences, test_labels):
    proposition_model, proposition_tokenizer = load_saved_model_and_tokenizer(PROPOSITION_MODEL_SAVE_PATH)

    worth_or_not_checking_model, worth_or_not_checking_tokenizer = load_saved_model_and_tokenizer(WORTH_CHECKING_PROPOSITIONS_MODEL_SAVE_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proposition_model.to(device)
    worth_or_not_checking_model.to(device)
    predicted_labels = []
    for sentence, label in zip(test_sentences, test_labels):
        inputs = proposition_tokenizer(sentence, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = proposition_model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        predicted_label = proposition_model.config.id2label[predicted_class_id]
        if predicted_label == 'not a factual proposition':
            predicted_labels.append(predicted_label)
        else:
            inputs = worth_or_not_checking_tokenizer(sentence, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = worth_or_not_checking_model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()
            predicted_label = worth_or_not_checking_model.config.id2label[predicted_class_id]
            predicted_labels.append(predicted_label)
    right_prediction_counter = 0
    for predicted_label, real_label in zip(predicted_labels, test_labels):
        if predicted_label == real_label:
            right_prediction_counter += 1

    accuracy = right_prediction_counter / len(test_labels)
    print(f"Test Accuracy by majority vote label: {accuracy:.4f}")



def create_two_phase_binary_tasks_models(sentences, labels):
    sentences, proposition_labels = split_dataset_to_factual_propositions_and_not_factual_propositions_labels(sentences, labels)
    # train_model(sentences, proposition_labels, model_save_path=PROPOSITION_MODEL_SAVE_PATH)
    print(f'factual proposition-not factual proposition model evaluation')
    evaluate_model(sentences, proposition_labels, saved_model_path=PROPOSITION_MODEL_SAVE_PATH)
    worth_or_not_checking_labels = []
    worth_or_not_checking_sentences = []
    for sentence, label in zip(sentences, labels):
        if label == 'worth checking' or label== 'not worth checking':
            worth_or_not_checking_sentences.append(sentence)
            worth_or_not_checking_labels.append(label)
    train_model(worth_or_not_checking_sentences, worth_or_not_checking_labels, model_save_path=WORTH_CHECKING_PROPOSITIONS_MODEL_SAVE_PATH)
    print(f'worth checking - not worth checking model evaluation')
    evaluate_model(worth_or_not_checking_sentences, worth_or_not_checking_labels, saved_model_path=WORTH_CHECKING_PROPOSITIONS_MODEL_SAVE_PATH)



if __name__ == '__main__':
    parsed_json_annotations_path = "D:\\data\\gili\\processed_knesset\\factuality_manual_annotations\\final_projects\\parsed_json_annotations"
    # sentences, labels = prepare_sentences_and_checkworthiness_labels_dataset(parsed_json_annotations_path)
    # save_sentences_labels(sentences,labels,path="sentences_checkworthiness_labels.json")
    sentences, labels = load_sentences_labels()
    # train_model(sentences, labels)

    # evaluate_model(sentences, labels)

    evaluate_model_on_sentences(test_sentences)

    # create_two_phase_binary_tasks_models(sentences, labels)
    # two_phase_models_evaluation(test_sentences, test_labels)
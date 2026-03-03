import logging
import json
from simpletransformers.question_answering import QuestionAnsweringModel


# ============================
# Logging (poți și să-l ștergi dacă vrei și mai simplu)
# ============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================
# Construim un mic dataset QA
# ============================
def build_toy_qa_data():
    """
    Returnează train_data și eval_data ca LISTE de dict-uri,
    formatul așteptat de SimpleTransformers.
    """

    # Exemplul 1
    context1 = "Paris is the capital city of France, located in Western Europe."
    qas1 = [
        {
            "id": "0",
            "is_impossible": False,
            "question": "What is the capital city of France?",
            "answers": [
                {
                    "text": "Paris",
                    "answer_start": context1.find("Paris"),
                }
            ],
        }
    ]

    # Exemplul 2
    context2 = (
        "The Danube is the second longest river in Europe, "
        "flowing through many countries."
    )
    qas2 = [
        {
            "id": "1",
            "is_impossible": False,
            "question": "Which river is the second longest in Europe?",
            "answers": [
                {
                    "text": "The Danube",
                    "answer_start": context2.find("The Danube"),
                }
            ],
        }
    ]

    train_data = [
        {"context": context1, "qas": qas1},
        {"context": context2, "qas": qas2},
    ]

    # Pentru demo, folosim același set și la evaluare
    eval_data = list(train_data)

    return train_data, eval_data


def main():
    # ----------------------------
    # Pregătim datele
    # ----------------------------
    train_data, eval_data = build_toy_qa_data()
    print("Exemplu de train_data (primul element):")
    print(train_data[0])

    # ----------------------------
    # Parametrii modelului
    # ----------------------------
    model_args = {
        "num_train_epochs": 3,
        "learning_rate": 3e-5,
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "max_seq_length": 128,
        "overwrite_output_dir": True,
        "reprocess_input_data": True,
        "output_dir": "qa_model_output",
        "save_steps": -1,
        "save_model_every_epoch": False,
        "evaluate_during_training": True,
        "evaluate_during_training_steps": 0,
        "logging_steps": 1,
    }

    # ----------------------------
    # Definim modelul QA (BERT)
    # ----------------------------
    model = QuestionAnsweringModel(
        "bert",
        "bert-base-uncased",
        args=model_args,
        use_cuda=False,  # pune True dacă ai GPU
    )

    # ----------------------------
    # Antrenare
    # ----------------------------
    print("\nÎncepem antrenarea QA...")
    model.train_model(train_data, eval_data=eval_data)

    # ----------------------------
    # Evaluare
    # ----------------------------
    print("\nEvaluăm modelul QA...")

    eval_output = model.eval_model(eval_data)
    # Poate întoarce fie result, fie (result, model_outputs)
    if isinstance(eval_output, tuple):
        result = eval_output[0]
    else:
        result = eval_output

    print("Eval result:", result)

    # ----------------------------
    # Predicție demo
    # ----------------------------
    prediction_data = [
        {
            "context": "Paris is the capital city of France.",
            "qas": [
                {
                    "id": "test-0",
                    "is_impossible": False,
                    "question": "What is the capital city of France?",
                }
            ],
        }
    ]

    print("\nRulăm o predicție...")
    answers, probs = model.predict(prediction_data)
    print("Răspunsuri prezise:", answers)
    print("Probabilități:", probs)

    # ----------------------------
    # Salvăm TOTUL într-un singur fișier
    # ----------------------------
    all_results = {
        "eval_result": result,
        "prediction_answers": answers,
        "prediction_probabilities": probs,
    }

    with open("qa_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nAm salvat toate rezultatele în qa_results.json")
    print("\nPentru laborator poți arăta în consolă:")
    print("  * log-urile de training (loss)")
    print("  * Eval result")
    print("  * Răspunsul prezis și probabilitatea lui")


if __name__ == "__main__":
    main()

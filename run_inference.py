import os
import json
import sys
import traceback
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def generate_paraphrases(text, model, tokenizer, num_return_sequences=3, max_length=128):
    """
    Generate paraphrases for the input text
    """
    try:
        input_ids = tokenizer(
            f"paraphrase: {text}",
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).input_ids

        outputs = model.generate(
            input_ids,
            num_return_sequences=num_return_sequences,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.1
        )

        paraphrases = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return paraphrases

    except Exception as e:
        print(f"Error generating paraphrases: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise

def main():
    print("Starting paraphrase generation", file=sys.stderr, flush=True)

    text = os.environ.get('INPUT_TEXT', 'Default input text')
    model_directory = os.environ.get('MODEL_DIRECTORY', '/model')

    output = {
        'input_text': text,
        'status': 'error',
        'paraphrases': []
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_directory)

        paraphrases = generate_paraphrases(text, model, tokenizer)
        output.update({
            'status': 'success',
            'paraphrases': paraphrases
        })

        for i, para in enumerate(paraphrases, 1):
            print(f"{i}. {para}", file=sys.stderr, flush=True)

    except Exception as e:
        print("Error during processing:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        output['error'] = str(e)

    os.makedirs('/outputs', exist_ok=True)
    output_path = '/outputs/result.json'

    try:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Successfully wrote output to {output_path}", file=sys.stderr, flush=True)
    except Exception as write_error:
        print("Error writing output file:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    main()

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
        # Prepare input
        input_ids = tokenizer(
            f"paraphrase: {text}", 
            return_tensors="pt", 
            max_length=max_length, 
            truncation=True
        ).input_ids
        
        # Generate paraphrases
        outputs = model.generate(
            input_ids, 
            num_return_sequences=num_return_sequences, 
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.1
        )
        
        # Decode generated paraphrases
        paraphrases = [
            tokenizer.decode(output, skip_special_tokens=True) 
            for output in outputs
        ]
        
        return paraphrases
    
    except Exception as e:
        print(f"Error generating paraphrases: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise

def main():
    # Print debugging information to stderr
    print("Starting paraphrase generation", file=sys.stderr, flush=True)
    print(f"Python version: {sys.version}", file=sys.stderr, flush=True)
    print(f"Torch version: {torch.__version__}", file=sys.stderr, flush=True)
    print(f"Torch device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}", file=sys.stderr, flush=True)
    
    # Get input text from environment variables
    text = os.environ.get('TEXT1', 'Default input text')
    
    # Prepare output dictionary
    output = {
        'input_text': text,
        'status': 'error',
        'paraphrases': []
    }
    
    try:
        # Model and tokenizer setup
        model_name = "prithivida/parrot_paraphraser_on_T5"
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Generate paraphrases
        paraphrases = generate_paraphrases(text, model, tokenizer)
        
        # Update output with successful result
        output.update({
            'status': 'success',
            'paraphrases': paraphrases
        })
        
        print("Generated paraphrases:", file=sys.stderr, flush=True)
        for i, para in enumerate(paraphrases, 1):
            print(f"{i}. {para}", file=sys.stderr, flush=True)
    
    except Exception as e:
        # Log the full error details to stderr
        print("Error during processing:", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        
        # Update output with error details
        output['error'] = str(e)
    
    # Ensure output directory exists
    output_dir = '/outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Write output to JSON file
    output_path = os.path.join(output_dir, 'result.json')
    
    try:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Successfully wrote output to {output_path}", file=sys.stderr, flush=True)
        print(f"File exists: {os.path.exists(output_path)}", file=sys.stderr, flush=True)
        print(f"File size: {os.path.getsize(output_path)} bytes", file=sys.stderr, flush=True)
    
    except Exception as write_error:
        print("Error writing output file:", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
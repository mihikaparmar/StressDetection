from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
import textwrap 

def text_summarizer_from_text(text):
    # Load BART model and tokenizer
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Tokenize and generate summary
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=550, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode and return the formatted summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    formatted_summary = "\n".join(textwrap.wrap(summary, width=80))  # Adjust width as needed
    return formatted_summary

# Example usage:
text_input = """
    Your text input here.
"""

summary = text_summarizer_from_text(text_input)
print("Summary:\n", summary)

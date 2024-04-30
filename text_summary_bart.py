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
In recent decades, technology has revolutionized the landscape of education, fundamentally altering the way knowledge is accessed, disseminated, and absorbed. 
The integration of technology into educational practices has catalyzed a paradigm shift, transcending the limitations of traditional pedagogical methods. 
One of the most profound impacts of technology on education is its role in fostering personalized learning experiences. 
Through adaptive learning platforms and intelligent tutoring systems, students can receive tailored instruction that caters to their individual learning styles and paces. 
Moreover, the proliferation of online resources and educational apps has democratized access to information, breaking down geographical barriers and providing learners from diverse backgrounds with equitable learning opportunities. 
Additionally, technology has redefined the dynamics of classroom interaction, facilitating collaborative learning through virtual forums, video conferencing tools, and cloud-based collaboration platforms. 
These digital platforms not only promote peer-to-peer engagement but also enable educators to transcend the confines of the traditional classroom, fostering global learning communities. 
However, while technology offers myriad benefits to education, it also presents challenges such as the digital divide, wherein disparities in access to technology exacerbate existing inequalities in education. 
Moreover, concerns regarding the overreliance on technology and its potential to impede critical thinking skills have been raised, highlighting the importance of striking a balance between technological integration and traditional pedagogy. 
Nevertheless, the transformative potential of technology in education is undeniable, offering educators and learners alike the tools to innovate, collaborate, and engage in lifelong learning endeavors. 
As technology continues to evolve, its impact on education will undoubtedly shape the future of learning, empowering individuals to navigate an increasingly complex and interconnected world with agility and resilience.
"""

# Now you can use the 'paragraph' variable in your Python code as needed


summary = text_summarizer_from_text(text_input)
print("Summary:\n", summary)

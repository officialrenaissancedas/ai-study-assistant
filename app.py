from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

app = Flask(__name__)

MODEL_NAME = "google/flan-t5-small"

# Load tokenizer & model with tie_word_embeddings=False to suppress warning
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, tie_word_embeddings=False)

# Text-generation pipeline
text_pipe = pipeline(
    task="text-generation",  # FIXED
    model=model,
    tokenizer=tokenizer
)

@app.route('/')
def home():
    return render_template('index.html')  # your frontend HTML file

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    text_input = data.get('text', '').strip()

    if not text_input:
        return jsonify({
            'summary': '⚠ Please enter some text!',
            'quiz': [],
            'concepts': [],
            'examples': [],
            'step_by_step': ''
        })

    try:
        # 1️⃣ Generate summary
        summary_prompt = f"Summarize this text clearly in points:\n{text_input}"
        summary_result = text_pipe(summary_prompt, max_new_tokens=200, do_sample=False)
        summary_text = summary_result[0].get('generated_text', '')

        # 2️⃣ Generate quiz
        quiz_prompt = f"Create 3 quiz questions based on this text:\n{text_input}"
        quiz_result = text_pipe(quiz_prompt, max_new_tokens=150, do_sample=False)
        quiz_text = quiz_result[0].get('generated_text', '')
        quiz_points = [q.strip('-•0123456789. ') for q in quiz_text.split('\n') if q.strip()]

        # 3️⃣ Generate key concepts
        concepts_prompt = f"List the key concepts and definitions in points:\n{text_input}"
        concepts_result = text_pipe(concepts_prompt, max_new_tokens=200, do_sample=False)
        concepts_text = concepts_result[0].get('generated_text', '')
        concepts_points = [c.strip('-•0123456789. ') for c in concepts_text.split('\n') if c.strip()]
        # Add placeholder images
        concepts_with_images = [{"title": c, "img_url": "https://via.placeholder.com/150"} for c in concepts_points]

        # 4️⃣ Generate examples
        examples_prompt = f"Provide simple examples and real-world applications:\n{text_input}"
        examples_result = text_pipe(examples_prompt, max_new_tokens=200, do_sample=False)
        examples_text = examples_result[0].get('generated_text', '')
        examples_points = [e.strip('-•0123456789. ') for e in examples_text.split('\n') if e.strip()]

        # 5️⃣ Step-by-step explanation
        step_prompt = f"Explain the content step by step:\n{text_input}"
        step_result = text_pipe(step_prompt, max_new_tokens=200, do_sample=False)
        step_text = step_result[0].get('generated_text', '')

        return jsonify({
            'summary': summary_text,
            'quiz': quiz_points,
            'concepts': concepts_with_images,
            'examples': examples_points,
            'step_by_step': step_text
        })

    except Exception as e:
        return jsonify({
            'summary': '⚠ Error generating output.',
            'quiz': [],
            'concepts': [],
            'examples': [],
            'step_by_step': str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

app = Flask(__name__)

MODEL_NAME = "google/flan-t5-small"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, tie_word_embeddings=False)

# Pipeline
text_pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer
)

@app.route('/')
def home():
    return render_template('index.html')

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
            'step_by_step': '',
            'eli5': '',
            'socratic': []
        })

    try:
        # Summary
        summary_prompt = f"Summarize this text clearly in points:\n{text_input}"
        summary_text = text_pipe(summary_prompt, max_new_tokens=200, do_sample=False)[0]['generated_text']

        # Quiz
        quiz_prompt = f"Create 3 quiz questions based on this text:\n{text_input}"
        quiz_text = text_pipe(quiz_prompt, max_new_tokens=150, do_sample=False)[0]['generated_text']
        quiz_points = [q.strip('-•0123456789. ') for q in quiz_text.split('\n') if q.strip()]

        # Concepts
        concepts_prompt = f"List key concepts and definitions:\n{text_input}"
        concepts_text = text_pipe(concepts_prompt, max_new_tokens=200, do_sample=False)[0]['generated_text']
        concepts_points = [c.strip('-•0123456789. ') for c in concepts_text.split('\n') if c.strip()]
        concepts_with_images = [{"title": c, "img_url": "https://via.placeholder.com/150"} for c in concepts_points]

        # Examples
        examples_prompt = f"Give real world examples:\n{text_input}"
        examples_text = text_pipe(examples_prompt, max_new_tokens=200, do_sample=False)[0]['generated_text']
        examples_points = [e.strip('-•0123456789. ') for e in examples_text.split('\n') if e.strip()]

        # Step-by-step
        step_prompt = f"Explain step by step:\n{text_input}"
        step_text = text_pipe(step_prompt, max_new_tokens=200, do_sample=False)[0]['generated_text']

        # 6️⃣ ELI5 Explanation
        eli5_prompt = f"Explain this in very simple terms like I'm 5 years old:\n{text_input}"
        eli5_result = text_pipe(eli5_prompt, max_new_tokens=150, do_sample=False)
        eli5_text = eli5_result[0].get('generated_text', '')
        # 7️⃣ Socratic Questions
        socratic_prompt = f"Ask 3 guiding Socratic questions to help understand this topic:\n{text_input}"
        socratic_result = text_pipe(socratic_prompt, max_new_tokens=150, do_sample=False)
        socratic_text = socratic_result[0].get('generated_text', '')
        socratic_questions = [q.strip('-•0123456789. ') for q in socratic_text.split('\n') if q.strip()]

        return jsonify({
            'summary': summary_text,
            'quiz': quiz_points,
            'concepts': concepts_with_images,
            'examples': examples_points,
            'step_by_step': step_text,
            'eli5': eli5_text,
            'socratic': socratic_questions
        })

    except Exception as e:
        return jsonify({
            'summary': '⚠ Error generating output.',
            'quiz': [],
            'concepts': [],
            'examples': [],
            'step_by_step': str(e),
            'eli5': '',
            'socratic': []
        })

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

app = Flask(__name__)

MODEL_NAME = "google/flan-t5-small"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# =========================================================
# ✅ FIX 1: CORRECT PIPELINE TASK (CRITICAL FIX)
# =========================================================
text_pipe = pipeline(
    task="text2text-generation",  # ✅ FIXED (was text-generation)
    model=model,
    tokenizer=tokenizer
)

# =========================================================
# ✅ FIX 2: SAFER CLEAN GENERATION
# =========================================================
def generate_clean(prompt, max_tokens=150):
    try:
        result = text_pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            max_length=None  # ✅ FIX warning
        )
        text = result[0]['generated_text']
        return text.replace(prompt, "").strip()
    except Exception as e:
        print("GEN ERROR:", str(e))  # ✅ DEBUG
        return "Error generating response"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    text_input = data.get('text', '').strip()

    print("GENERATE INPUT:", text_input)  # ✅ DEBUG

    if not text_input:
        return jsonify({
            'summary': '⚠ Please enter some text!',
            'quiz': [],
            'concepts': [],
            'examples': [],
            'step_by_step': '',
            'eli5': '',
            'socratic': [],
            'chat': ''
        })

    try:
        # ✅ CLEAN GENERATION (FIXED)
        summary_text = generate_clean(f"Summarize clearly in points:\n{text_input}", 200)

        quiz_text = generate_clean(f"Create 3 quiz questions:\n{text_input}", 150)
        quiz_points = [q.strip('-•0123456789. ') for q in quiz_text.split('\n') if q.strip()]

        concepts_text = generate_clean(f"List key concepts with short definitions:\n{text_input}", 200)
        concepts_points = [c.strip('-•0123456789. ') for c in concepts_text.split('\n') if c.strip()]
        concepts_with_images = [{"title": c, "img_url": "https://via.placeholder.com/150"} for c in concepts_points]

        examples_text = generate_clean(f"Give simple real-world examples:\n{text_input}", 200)
        examples_points = [e.strip('-•0123456789. ') for e in examples_text.split('\n') if e.strip()]

        step_text = generate_clean(f"Explain step-by-step clearly:\n{text_input}", 200)

        eli5_text = generate_clean(f"Explain like I'm 5 years old in very simple words:\n{text_input}", 150)

        socratic_text = generate_clean(f"Ask 3 guiding Socratic questions:\n{text_input}", 150)
        socratic_questions = [q.strip('-•0123456789. ') for q in socratic_text.split('\n') if q.strip()]

        # ✅ IMPROVED CHAT PROMPT (BETTER OUTPUT)
        chat_reply = generate_clean(f"Question: {text_input}\nAnswer:", 100)

        print("GENERATE SUCCESS")  # ✅ DEBUG

        return jsonify({
            'summary': summary_text,
            'quiz': quiz_points,
            'concepts': concepts_with_images,
            'examples': examples_points,
            'step_by_step': step_text,
            'eli5': eli5_text,
            'socratic': socratic_questions,
            'chat': chat_reply
        })

    except Exception as e:
        print("GENERATE ERROR:", str(e))  # ✅ DEBUG
        return jsonify({
            'summary': '⚠ Error generating output.',
            'quiz': [],
            'concepts': [],
            'examples': [],
            'step_by_step': str(e),
            'eli5': '',
            'socratic': [],
            'chat': 'Error generating response.'
        })

# =========================================================
# ✅ NEW CHAT ROUTE (ADDED — DOES NOT CHANGE ORIGINAL LOGIC)
# =========================================================
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()

        print("CHAT INPUT:", user_input)  # ✅ DEBUG

        if not user_input:
            return jsonify({"chat": "Please ask something!"})

        # ✅ IMPROVED PROMPT
        response = generate_clean(
            f"Question: {user_input}\nAnswer:",
            100
        )

        print("CHAT OUTPUT:", response)  # ✅ DEBUG

        return jsonify({"chat": response})

    except Exception as e:
        print("CHAT ERROR:", str(e))  # ✅ DEBUG
        return jsonify({"chat": "Server error occurred"})

# =========================================================

if __name__ == "__main__":
    app.run(debug=True)
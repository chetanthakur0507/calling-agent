import os
from collections import defaultdict

from flask import Flask, abort, request
from openai import OpenAI
from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import Gather, VoiceResponse


def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()

app = Flask(__name__)

GROK_API_KEY = os.getenv("GROK_API_KEY", "").strip()
GROK_MODEL = os.getenv("GROK_MODEL", "grok-2-latest")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()

if not GROK_API_KEY:
    raise RuntimeError("GROK_API_KEY missing. Add it in .env file.")

client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")

# In-memory chat context per call. Good for local testing.
conversation_memory = defaultdict(list)


@app.before_request
def validate_twilio_request():
    """Validate webhook signature if TWILIO_AUTH_TOKEN is provided."""
    if not TWILIO_AUTH_TOKEN:
        return

    signature = request.headers.get("X-Twilio-Signature", "")
    validator = RequestValidator(TWILIO_AUTH_TOKEN)
    is_valid = validator.validate(request.url, request.form, signature)
    if not is_valid:
        abort(403)


def ask_grok(call_sid: str, user_text: str) -> str:
    history = conversation_memory[call_sid]

    messages = [
        {
            "role": "system",
            "content": (
                "Tum ek friendly Hindi/Hinglish phone assistant ho. "
                "Short, clear, polite replies do. "
                "Agar user unclear ho to ek follow-up question poochho."
            ),
        }
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    completion = client.chat.completions.create(
        model=GROK_MODEL,
        messages=messages,
        temperature=0.6,
        max_tokens=180,
    )

    ai_text = completion.choices[0].message.content.strip()

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": ai_text})

    # Keep memory bounded
    if len(history) > 12:
        conversation_memory[call_sid] = history[-12:]

    return ai_text


def build_gather(prompt: str) -> Gather:
    gather = Gather(
        input="speech",
        action="/process",
        method="POST",
        language="hi-IN",
        speech_timeout="auto",
        hints="Hindi,Hinglish,English,help,price,college,anonymous",
    )
    gather.say(prompt, language="hi-IN")
    return gather


@app.route("/health", methods=["GET"])
def health():
    return {"ok": True}, 200


@app.route("/", methods=["GET"])
def index():
    return (
        "<h1>Calling Agent is running</h1>"
        "<p>Use the Twilio webhook URL: <code>/voice</code></p>"
        "<p>Example: <code>https://your-ngrok-url.ngrok-free.app/voice</code></p>"
    ), 200


@app.route("/voice", methods=["POST"])
def voice():
    call_sid = request.form.get("CallSid", "unknown")
    conversation_memory[call_sid] = []

    vr = VoiceResponse()
    vr.append(
        build_gather(
            "Namaste, aap AI calling assistant se baat kar rahe hain. "
            "Boliye, aapko kis cheez mein madad chahiye?"
        )
    )

    # If speech capture fails, Twilio continues to next verb.
    vr.say("Mujhe awaaz clear nahi mili. Kripya dobara call karein.", language="hi-IN")
    vr.hangup()

    return str(vr)


@app.route("/process", methods=["POST"])
def process():
    call_sid = request.form.get("CallSid", "unknown")
    user_text = (request.form.get("SpeechResult") or "").strip()

    vr = VoiceResponse()

    if not user_text:
        vr.append(
            build_gather(
                "Mujhe aapki baat sahi se samajh nahi aayi. "
                "Kripya thoda clearly dobara boliye."
            )
        )
        return str(vr)

    try:
        ai_reply = ask_grok(call_sid, user_text)
    except Exception:
        ai_reply = "Sorry, abhi server par problem aa rahi hai. Kripya thodi der baad try karein."

    vr.say(ai_reply, language="hi-IN")

    # Keep conversation going naturally.
    vr.append(build_gather("Agar aur koi sawal hai to ab boliye."))
    vr.say("Call karne ke liye dhanyavaad. Namaste.", language="hi-IN")
    vr.hangup()

    return str(vr)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

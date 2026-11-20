import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

app = FastAPI(title="Poetic Voice Diary")

LISTENER_PROMPT = (
    "You are a warm, caring friend talking to someone about their day. "
    "Ask ONE short question at a time. React with empathy to what they say. "
    "Keep replies under 2 sentences, natural and spoken-style, no lists, no emojis."
)

POET_PROMPT = (
    "You are a gentle poet. Read this conversation about someone's day and write "
    "a short poem diary entry (8-16 lines) in first person, as if the person wrote it. "
    "Use soft imagery and capture their real moments and feelings. "
    "Give it a title. Return only the poem."
)

class Message(BaseModel):
    role: str      # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

@app.get("/")
def root():
    return {"status": "Poetic Voice Diary is running"}

@app.post("/chat")
def chat(req: ChatRequest):
    msgs = [{"role": "system", "content": LISTENER_PROMPT}]
    msgs += [m.model_dump() for m in req.messages]
    r = client.chat.completions.create(
        model=MODEL, messages=msgs, temperature=0.8, max_tokens=150
    )
    return {"reply": r.choices[0].message.content}

@app.post("/poem")
def poem(req: ChatRequest):
    convo = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": POET_PROMPT},
            {"role": "user", "content": convo},
        ],
        temperature=0.9,
        max_tokens=400,
    )
    return {"poem": r.choices[0].message.content}

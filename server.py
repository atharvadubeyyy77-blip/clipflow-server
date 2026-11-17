import os, uuid, json, shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DATA = Path(os.getenv("DATA_DIR", "data"))
DATA.mkdir(exist_ok=True)
app = FastAPI(title="ClipFlow server")
_model = None


def whisper():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(os.getenv("WHISPER_MODEL", "tiny"), device="cpu", compute_type="int8")
    return _model


def path_for(vid: str) -> Path:
    if not vid.isalnum():
        raise HTTPException(400, "bad id")
    p = DATA / f"{vid}.mp4"
    if not p.exists():
        raise HTTPException(404, "video not found")
    return p


@app.get("/api/health")
def health():
    return {"ok": True, "ai": bool(os.getenv("ANTHROPIC_API_KEY"))}


class ImportReq(BaseModel):
    url: str


@app.post("/api/import")
def import_video(req: ImportReq):
    import yt_dlp
    vid = uuid.uuid4().hex[:12]
    opts = {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/b",
        "outtmpl": str(DATA / f"{vid}.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "max_filesize": 2_000_000_000,
    }
    if os.getenv("YT_COOKIES"):
        opts["cookiefile"] = os.getenv("YT_COOKIES")
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(req.url, download=True)
    except Exception as e:
        raise HTTPException(400, f"Could not download: {str(e)[:200]}")
    if not (DATA / f"{vid}.mp4").exists():
        raise HTTPException(500, "Download finished but no mp4 was produced")
    return {"id": vid, "title": info.get("title", "video"), "heatmap": info.get("heatmap")}


@app.get("/api/media/{vid}")
def media(vid: str):
    return FileResponse(path_for(vid), media_type="video/mp4")


@app.post("/api/transcribe")
def transcribe(file: UploadFile | None = File(None), id: str | None = Form(None)):
    tmp = None
    if id:
        src = path_for(id)
    elif file:
        tmp = DATA / f"up_{uuid.uuid4().hex[:10]}"
        with open(tmp, "wb") as f:
            shutil.copyfileobj(file.file, f)
        src = tmp
    else:
        raise HTTPException(400, "send a file or an id")
    try:
        segs, _ = whisper().transcribe(str(src), word_timestamps=True, vad_filter=True)
        out = [
            {
                "start": s.start, "end": s.end, "text": s.text.strip(),
                "words": [{"word": w.word, "start": w.start, "end": w.end} for w in (s.words or [])],
            }
            for s in segs
        ]
    finally:
        if tmp:
            tmp.unlink(missing_ok=True)
    return {"segments": out}


class Seg(BaseModel):
    start: float
    end: float
    text: str


class MomentReq(BaseModel):
    segments: list[Seg]
    length: int = 30
    count: int = 5
    start: float = 0
    end: float = 0
    heatmap: list[dict] | None = None


@app.post("/api/moments")
def moments(req: MomentReq):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(400, "Set ANTHROPIC_API_KEY on the server")
    import anthropic
    transcript = "\n".join(f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in req.segments)[:150000]
    hm = sorted(req.heatmap or [], key=lambda h: -h.get("value", 0))[:12]
    hint = ""
    if hm:
        hint = "\nMost-replayed moments on YouTube (start-end: strength): " + ", ".join(
            f"{h['start_time']:.0f}-{h['end_time']:.0f}: {h['value']:.2f}" for h in hm)
    prompt = f"""You are a short-form video editor for TikTok, Reels and Shorts.
From the transcript, pick the {req.count} best standalone clips, each about {req.length} seconds (never under {int(req.length * 0.6)} or over {int(req.length * 1.5)}).
Prefer: a strong first sentence that hooks, a complete idea or story, emotion, surprise, strong opinions, useful tips, and a clear payoff.
Avoid intros, ads, filler, and cutting mid-sentence. Clips must not overlap and must start and end on segment boundaries.
For each clip give a punchy hook title (max 8 words, honest, not misleading), a one-line reason, and a 0-100 virality score.
Reply with ONLY JSON: {{"moments":[{{"start":0.0,"end":0.0,"title":"hook","why":"reason","score":0}}]}}
{hint}

Transcript:
{transcript}"""
    msg = anthropic.Anthropic(api_key=key).messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5"), max_tokens=2000,
        messages=[{"role": "user", "content": prompt}])
    text = msg.content[0].text
    try:
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        raise HTTPException(502, "AI returned invalid JSON, try again")
    hi = (req.end or 1e9) + 1
    ms = [m for m in data.get("moments", []) if req.start <= m["start"] < m["end"] <= hi]
    return {"moments": ms[: req.count]}


app.mount("/", StaticFiles(directory="static", html=True), name="static")

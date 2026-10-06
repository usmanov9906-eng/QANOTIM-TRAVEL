import asyncio
import json
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import psycopg
from aiohttp import web
from psycopg.rows import dict_row
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandObject
from aiogram.types import (BufferedInputFile, CallbackQuery,
                           InlineKeyboardButton, InlineKeyboardMarkup,
                           InputMediaPhoto, Message)
from anthropic import AsyncAnthropic
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("qanotim")

# ---------- SOZLAMALAR (Koyeb Environment variables) ----------
BOT_TOKEN = os.environ["BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")   # ixtiyoriy (pullik)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")         # bepul: aistudio.google.com
GEMINI_MODEL = os.environ.get("GEMINI_MODEL")   # bo'sh qolsa bot o'zi mos modelni topadi
ADMIN_ID = int(os.environ["ADMIN_ID"])            # sizning Telegram ID raqamingiz
_ch = os.environ["CHANNEL_ID"]                    # @qanotim_travel yoki -100...
CHANNEL_ID = int(_ch) if _ch.lstrip("-").isdigit() else _ch
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
SOURCE = os.environ.get("SOURCE_CHANNEL", "asialuxetraveluzb")
DATABASE_URL = os.environ["DATABASE_URL"]   # Neon/Supabase Postgres manzili
TZ = ZoneInfo("Asia/Tashkent")
POLL_SECONDS = 600
# soat -> post turi (3 ta qaynoq tur + 1 ta viza)
SLOTS = {10: "tour", 13: "tour", 16: "visa", 19: "tour"}

FOOTER = (
    "QANOTIM TRAVEL — саёҳат агентлиги ва виза маркази ✈️\n\n"
    "📞 +998 33 699 95 95 | @QANOTIM_TRAVEL\n"
    "📷 instagram.com/qanotim_uz\n"
    "📍 Наманган, Уйчи кўчаси 243."
)

SYSTEM = """Sen sayohat agentligi uchun tarjimon va muharrirsan.
Senga rus tilida qaynoq tur e'loni beriladi. Uni o'zbek tiliga KIRILL alifbosida tarjima qil va
reklama uslubida chiroyli, qisqa tahrir qil.
Qoidalar:
- Narxlar, sanalar, yo'nalish, mehmonxona nomlari, raqamlar va emojilar aynan saqlansin. Hech narsani o'zing o'ylab topma.
- Asialuxe Travel yoki boshqa operatorning telefon raqami, @username, havolalari va nomi umuman olib tashlansin.
- Oxiriga aloqa ma'lumoti QO'SHMA, uni dastur o'zi qo'shadi.
- Faqat tayyor post matnini qaytar, izoh yozma.
- Agar matn tur/viza e'loni emas (ichki xabar, e'lon, tabrik va h.k.) bo'lsa, faqat SKIP so'zini qaytar."""

claude = AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
bot = Bot(BOT_TOKEN)
dp = Dispatcher()
editing: dict[int, int] = {}   # admin -> tahrirlanayotgan post id


# ---------- BAZA (Postgres, doimiy saqlanadi) ----------
def _conn():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row, autocommit=True)


def execute(sql, args=()):
    sql = sql.replace("?", "%s")
    is_post_insert = sql.lstrip().upper().startswith("INSERT INTO POSTS")
    if is_post_insert:
        sql += " RETURNING id"
    with _conn() as c:
        cur = c.execute(sql, args)
        return cur.fetchone()["id"] if is_post_insert else None


def select(sql, args=()):
    with _conn() as c:
        return c.execute(sql.replace("?", "%s"), args).fetchall()


def init_db():
    execute("""CREATE TABLE IF NOT EXISTS posts(
        id SERIAL PRIMARY KEY,
        src_id TEXT UNIQUE, kind TEXT, status TEXT,
        text TEXT, photos TEXT, videos TEXT, last_pub TEXT)""")
    execute("CREATE TABLE IF NOT EXISTS slots(key TEXT PRIMARY KEY)")


# ---------- MANBA KANALNI O'QISH ----------
def fetch_source():
    html = httpx.get(f"https://t.me/s/{SOURCE}", timeout=20,
                     follow_redirects=True).text
    soup = BeautifulSoup(html, "html.parser")
    posts = []
    for m in soup.select(".tgme_widget_message"):
        if not m.get("data-post"):
            continue
        t = m.select_one(".tgme_widget_message_text")
        photos = []
        for a in m.select(".tgme_widget_message_photo_wrap"):
            style = a.get("style", "")
            if "url('" in style:
                photos.append(style.split("url('")[1].split("')")[0])
        videos = [v["src"] for v in m.select("video") if v.get("src")]
        posts.append({"src_id": m["data-post"],
                      "text": t.get_text("\n").strip() if t else "",
                      "photos": photos, "videos": videos})
    return posts


_gem_model = GEMINI_MODEL
GEM_BASE = "https://generativelanguage.googleapis.com/v1beta"


async def gemini_pick_model(c, exclude=()):
    """Kalitingiz uchun mavjud modellar ichidan mosini tanlaydi."""
    global _gem_model
    r = await c.get(f"{GEM_BASE}/models?pageSize=200",
                    headers={"x-goog-api-key": GEMINI_API_KEY})
    r.raise_for_status()
    names = [m["name"].split("/", 1)[1] for m in r.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    log.info("Gemini modellari: %s", names)
    prefer = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-flash-lite-latest",
              "gemini-2.5-flash-lite", "gemini-2.0-flash"]
    for p in prefer:
        if p in names and p not in exclude:
            _gem_model = p
            log.info("Tanlangan Gemini modeli: %s", p)
            return
    bad = ("image", "tts", "live", "audio", "exp", "thinking", "embedding", "robotics")
    flash = [n for n in names if "flash" in n and n not in exclude
             and not any(x in n for x in bad)]
    if not flash:
        raise RuntimeError("Mos Gemini modeli topilmadi")
    _gem_model = flash[0]
    log.info("Tanlangan Gemini modeli: %s", _gem_model)


async def translate(text: str) -> str:
    global _gem_model
    if GEMINI_API_KEY:
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"temperature": 0.3},
        }
        async with httpx.AsyncClient(timeout=90) as c:
            if not _gem_model:
                await gemini_pick_model(c)
            r = None
            for attempt in (1, 2):
                r = await c.post(f"{GEM_BASE}/models/{_gem_model}:generateContent",
                                 json=body,
                                 headers={"x-goog-api-key": GEMINI_API_KEY})
                if r.status_code == 404 and attempt == 1:
                    failed = _gem_model
                    await gemini_pick_model(c, exclude=(failed,))
                    continue
                break
            r.raise_for_status()
            data = r.json()
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    if claude:
        r = await claude.messages.create(
            model=MODEL, max_tokens=1500, system=SYSTEM,
            messages=[{"role": "user", "content": text}])
        return r.content[0].text.strip()
    raise RuntimeError("GEMINI_API_KEY yoki ANTHROPIC_API_KEY kerak")


# ---------- POST YUBORISH ----------
async def download(url: str):
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
            r = await c.get(url)
            if r.status_code == 200 and len(r.content) < 49_000_000:
                return r.content
    except Exception as e:
        log.warning("download xato: %s", e)
    return None


async def send_post(chat_id, row):
    text = row["text"] + "\n\n" + FOOTER
    photos = json.loads(row["photos"] or "[]")
    videos = json.loads(row["videos"] or "[]")
    cap = text if len(text) <= 1024 else None
    sent = False

    if videos:
        data = await download(videos[0])
        if data:
            await bot.send_video(chat_id, BufferedInputFile(data, "video.mp4"),
                                 caption=cap)
            sent = True
    if not sent and photos:
        files = [await download(u) for u in photos[:10]]
        files = [f for f in files if f]
        if len(files) == 1:
            await bot.send_photo(chat_id, BufferedInputFile(files[0], "1.jpg"),
                                 caption=cap)
            sent = True
        elif files:
            media = [InputMediaPhoto(media=BufferedInputFile(f, f"{i}.jpg"),
                                     caption=cap if i == 0 else None)
                     for i, f in enumerate(files)]
            await bot.send_media_group(chat_id, media)
            sent = True
    if not sent or cap is None:
        await bot.send_message(chat_id, text)


def buttons(pid):
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Tasdiqlash", callback_data=f"ok:{pid}"),
        InlineKeyboardButton(text="✏️ Tahrirlash", callback_data=f"ed:{pid}"),
        InlineKeyboardButton(text="❌ Rad etish", callback_data=f"no:{pid}"),
    ]])


async def show_for_review(pid):
    row = select("SELECT * FROM posts WHERE id=?", (pid,))[0]
    await send_post(ADMIN_ID, row)
    await bot.send_message(ADMIN_ID, f"Post #{pid}. Kanalga qo'yamizmi?",
                           reply_markup=buttons(pid))


# ---------- AVTOMATIK JARAYONLAR ----------
async def poll_loop():
    while True:
        try:
            posts = await asyncio.to_thread(fetch_source)
            first_run = not select(
                "SELECT 1 FROM posts WHERE src_id IS NOT NULL LIMIT 1")
            for i, p in enumerate(posts):
                if select("SELECT 1 FROM posts WHERE src_id=?", (p["src_id"],)):
                    continue
                old = first_run and i < len(posts) - 5   # birinchi ishga tushganda faqat oxirgi 5 tasi
                if old or not p["text"]:
                    execute("INSERT INTO posts(src_id,kind,status,text,photos,videos)"
                            " VALUES(?,?,?,?,?,?)",
                            (p["src_id"], "tour", "skipped", "", "[]", "[]"))
                    continue
                try:
                    uz = await translate(p["text"])
                except Exception as e:
                    log.warning("tarjima xato (%s): %s", p["src_id"], e)
                    await asyncio.sleep(5)
                    continue   # keyingi tekshiruvda qayta uriniladi
                await asyncio.sleep(4)   # bepul limitga urilmaslik uchun
                if uz.strip().upper() == "SKIP":
                    execute("INSERT INTO posts(src_id,kind,status,text,photos,videos)"
                            " VALUES(?,?,?,?,?,?)",
                            (p["src_id"], "tour", "skipped", "", "[]", "[]"))
                    continue
                pid = execute(
                    "INSERT INTO posts(src_id,kind,status,text,photos,videos)"
                    " VALUES(?,?,?,?,?,?)",
                    (p["src_id"], "tour", "pending", uz,
                     json.dumps(p["photos"]), json.dumps(p["videos"])))
                await show_for_review(pid)
        except Exception as e:
            log.exception("poll xato: %s", e)
        await asyncio.sleep(POLL_SECONDS)


async def publish(kind):
    if kind == "tour":
        rows = select("SELECT * FROM posts WHERE kind='tour' AND status='queued'"
                      " ORDER BY id LIMIT 1")
    else:
        rows = select("SELECT * FROM posts WHERE kind='visa' AND status='queued'"
                      " ORDER BY COALESCE(last_pub,'') LIMIT 1")
    if not rows:
        await bot.send_message(ADMIN_ID, f"⚠️ {kind} navbati bo'sh, bu soatda post chiqmadi.")
        return
    row = rows[0]
    await send_post(CHANNEL_ID, row)
    if kind == "tour":
        execute("UPDATE posts SET status='published' WHERE id=?", (row["id"],))
    else:
        execute("UPDATE posts SET last_pub=? WHERE id=?",
                (datetime.now(TZ).isoformat(), row["id"]))


async def schedule_loop():
    while True:
        now = datetime.now(TZ)
        kind = SLOTS.get(now.hour)
        key = now.strftime("%Y-%m-%d-%H")
        if kind and not select("SELECT 1 FROM slots WHERE key=?", (key,)):
            execute("INSERT INTO slots(key) VALUES(?)", (key,))
            try:
                await publish(kind)
            except Exception as e:
                log.exception("publish xato: %s", e)
        await asyncio.sleep(30)


# ---------- ADMIN BUYRUQLARI ----------
@dp.message(Command("start"), F.from_user.id == ADMIN_ID)
async def start(m: Message):
    await m.answer("Bot ishlayapti.\n/queue — navbat holati\n"
                   "/addviza matn — viza posti qo'shish (har safar navbat bilan aylanadi)")


@dp.message(Command("queue"), F.from_user.id == ADMIN_ID)
async def queue(m: Message):
    t = select("SELECT COUNT(*) n FROM posts WHERE kind='tour' AND status='queued'")[0]["n"]
    v = select("SELECT COUNT(*) n FROM posts WHERE kind='visa' AND status='queued'")[0]["n"]
    p = select("SELECT COUNT(*) n FROM posts WHERE status='pending'")[0]["n"]
    await m.answer(f"Navbatda: {t} tur, {v} viza shabloni.\nTasdiq kutayotgan: {p}")


@dp.message(Command("addviza"), F.from_user.id == ADMIN_ID)
async def addviza(m: Message, command: CommandObject):
    if not command.args:
        await m.answer("Ishlatilishi: /addviza viza posti matni (kontaktsiz, ular o'zi qo'shiladi)")
        return
    execute("INSERT INTO posts(kind,status,text,photos,videos) VALUES('visa','queued',?,'[]','[]')",
            (command.args,))
    await m.answer("✅ Viza posti navbatga qo'shildi.")


@dp.callback_query(F.data.startswith("ok:"), F.from_user.id == ADMIN_ID)
async def ok(cb: CallbackQuery):
    execute("UPDATE posts SET status='queued' WHERE id=?", (int(cb.data[3:]),))
    await cb.message.edit_text("✅ Navbatga qo'yildi.")


@dp.callback_query(F.data.startswith("no:"), F.from_user.id == ADMIN_ID)
async def no(cb: CallbackQuery):
    execute("UPDATE posts SET status='rejected' WHERE id=?", (int(cb.data[3:]),))
    await cb.message.edit_text("❌ Rad etildi.")


@dp.callback_query(F.data.startswith("ed:"), F.from_user.id == ADMIN_ID)
async def ed(cb: CallbackQuery):
    editing[ADMIN_ID] = int(cb.data[3:])
    await cb.message.edit_text("✏️ Yangi matnni yuboring (kontaktsiz, ular o'zi qo'shiladi).")


@dp.message(F.text, ~F.text.startswith("/"), F.from_user.id == ADMIN_ID)
async def new_text(m: Message):
    pid = editing.pop(ADMIN_ID, None)
    if not pid:
        return
    execute("UPDATE posts SET text=? WHERE id=?", (m.text, pid))
    await show_for_review(pid)


async def start_health_server():
    """Render bepul web service uxlab qolmasligi uchun kichik sahifa (PORT bo'lsa ishlaydi)."""
    port = os.environ.get("PORT")
    if not port:
        return
    app = web.Application()
    app.router.add_get("/", lambda request: web.Response(text="ok"))
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(port)).start()


async def main():
    init_db()
    await start_health_server()
    asyncio.create_task(poll_loop())
    asyncio.create_task(schedule_loop())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

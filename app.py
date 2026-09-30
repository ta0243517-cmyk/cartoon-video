import glob, hashlib, json, math, os, shutil, subprocess, tempfile, time, wave
import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Cartoon Video", layout="centered")

MAX_MIN = 30      # longest video allowed (minutes)
CHUNK = 60        # seconds per part (saved separately, so work can resume)
SR = 22050        # sound-effect sample rate
RATIOS = {
    "Same as original": None,
    "YouTube / Facebook / X (16:9)": (16, 9),
    "TikTok / Reels / Shorts (9:16)": (9, 16),
    "Instagram square (1:1)": (1, 1),
    "Instagram / Facebook portrait (4:5)": (4, 5),
    "Classic (4:3)": (4, 3),
}
FITS = ["Crop (fill screen)", "Fit (black bars)"]
ZOOMS = ["Off", "Slow zoom in/out (each shot)", "Punch zoom on scene changes",
         "Pulse zoom on loud moments"]
SENS = {"Low (fewer cuts)": 40, "Normal": 28, "High (more cuts)": 18}
TMP = tempfile.gettempdir()


# ---------------- video helpers ----------------
def probe(p):
    o = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate:stream_tags=rotate:stream_side_data=rotation:format=duration",
         "-of", "json", p], capture_output=True, text=True).stdout
    j = json.loads(o)
    s = j["streams"][0]
    n, d = s["avg_frame_rate"].split("/")
    fps = float(n) / float(d) if float(d) else 25.0
    rot = 0
    for sd in s.get("side_data_list", []):
        rot = sd.get("rotation", rot)
    rot = int(float(s.get("tags", {}).get("rotate", rot)))
    w, h = s["width"], s["height"]
    if abs(rot) % 180 == 90:
        w, h = h, w
    return w, h, fps, float(j["format"]["duration"])


def has_audio(p):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                        "stream=index", "-of", "csv=p=0", p], capture_output=True, text=True).stdout
    return bool(o.strip())


def dims(ratio, q):
    rw, rh = ratio
    w, h = (q * rw / rh, q) if rw >= rh else (q, q * rh / rw)
    return int(w) // 2 * 2, int(h) // 2 * 2


def vf_for(W, H, fit):
    if fit.startswith("Crop"):
        return f"scale={W}:{H}:flags=lanczos:force_original_aspect_ratio=increase,crop={W}:{H}"
    return (f"scale={W}:{H}:flags=lanczos:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2")


def cartoon(img, line, smooth, levels):
    h, w = img.shape[:2]
    s = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    for _ in range(smooth):
        s = cv2.bilateralFilter(s, 7, 40, 7)
    color = cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    step = 256 // levels
    color = (color // step) * step + step // 2
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 4)
    if line > 1:
        edges = cv2.erode(edges, np.ones((line, line), np.uint8))
    return cv2.bitwise_and(color, color, mask=edges)


def grab_frame(src, t, W, H, fit):
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", src, "-frames:v", "1",
         "-vf", vf_for(W, H, fit), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True)
    return np.frombuffer(r.stdout, np.uint8)[:W * H * 3].reshape(H, W, 3).copy()


# ---------------- zoom ----------------
def zoom_frame(fr, z):
    if z <= 1.002:
        return fr
    h, w = fr.shape[:2]
    nw, nh = int(w / z) // 2 * 2, int(h / z) // 2 * 2
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    return cv2.resize(fr[y0:y0 + nh, x0:x0 + nw], (w, h), interpolation=cv2.INTER_CUBIC)


def spaced(times, gap):
    out = []
    for t in times:
        if not out or t - out[-1] >= gap:
            out.append(t)
    return out


def make_zoom(mode, amt, cuts, peaks, dur):
    if mode == "Off":
        return None
    if mode.startswith("Pulse"):
        pk = np.array(spaced(peaks, 1.5))
        if not len(pk):
            return None

        def f(t):
            i = np.searchsorted(pk, t, side="right") - 1
            return 1.0 if i < 0 else 1 + amt * math.exp(-(t - pk[i]) * 7)
        return f
    real = np.array(sorted(set([0.0] + list(cuts))))
    if mode.startswith("Punch"):
        def f(t):
            i = np.searchsorted(real, t, side="right") - 1
            return 1 + amt * math.exp(-(t - real[i]) * 6)
        return f
    starts = []                                   # slow zoom: also split very long shots
    bounds = list(real) + [dur]
    for a, b in zip(bounds[:-1], bounds[1:]):
        starts.append(a)
        x = a + 10
        while x < b - 2:
            starts.append(x); x += 10
    arr = np.array(sorted(starts))
    ends = list(arr[1:]) + [dur]

    def f(t):
        i = np.searchsorted(arr, t, side="right") - 1
        L = max(min(ends[i] - arr[i], 10.0), 1.0)
        p = min(max((t - arr[i]) / L, 0.0), 1.0)
        p = p * p * (3 - 2 * p)
        return 1 + amt * (p if i % 2 == 0 else 1 - p)
    return f


# ---------------- analysis (scene changes, loud moments) ----------------
def get_cuts(src, wd, thr, key):
    path = os.path.join(wd, f"cuts_{key}.json")
    if os.path.exists(path):
        return json.load(open(path))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-vf",
                          "fps=8,scale=160:90,format=gray", "-f", "rawvideo", "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fs, prev, i, cuts = 160 * 90, None, 0, []
    while True:
        b = p.stdout.read(fs)
        if len(b) < fs:
            break
        g = np.frombuffer(b, np.uint8).astype(np.int16)
        if prev is not None and np.abs(g - prev).mean() > thr:
            t = round(i / 8, 2)
            if not cuts or t - cuts[-1] >= 0.8:
                cuts.append(t)
        prev, i = g, i + 1
    p.wait()
    json.dump(cuts, open(path, "w"))
    return cuts


def get_peaks(src, wd):
    path = os.path.join(wd, "peaks.json")
    if os.path.exists(path):
        return json.load(open(path))
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", "8000",
                        "-f", "s16le", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, np.int16).astype(np.float32)
    peaks = []
    if a.size >= 16000:
        m = a.size // 800
        e = np.sqrt((a[:m * 800].reshape(m, 800) ** 2).mean(1))
        d = e - np.convolve(e, np.ones(10) / 10, mode="same")
        thr = np.percentile(d, 96)
        for i in range(1, m - 1):
            if d[i] > thr and d[i] >= d[i - 1] and d[i] >= d[i + 1]:
                peaks.append(round(i * 0.1, 2))
        peaks = spaced(peaks, 1.0)
    json.dump(peaks, open(path, "w"))
    return peaks


# ---------------- sound effects (made by code, no copyright) ----------------
def make_sfx():
    rng = np.random.default_rng(1)
    n = int(0.6 * SR); t = np.linspace(0, 1, n)
    noise = rng.standard_normal(n).astype(np.float32)
    coef = 0.02 + 0.5 * np.sin(np.pi * t) ** 2
    y, acc = np.zeros(n, np.float32), 0.0
    for i in range(n):
        acc += coef[i] * (noise[i] - acc); y[i] = acc
    y = y * np.sin(np.pi * t) ** 1.5
    whoosh = y / max(np.abs(y).max(), 1e-6)
    n = int(0.7 * SR); tt = np.arange(n) / SR
    ph = 2 * np.pi * np.cumsum(50 + 110 * np.exp(-tt * 12)) / SR
    hit = np.sin(ph) * np.exp(-tt * 6) + 0.25 * rng.standard_normal(n) * np.exp(-tt * 40)
    hit = (hit / np.abs(hit).max()).astype(np.float32)
    n = int(0.15 * SR); tt = np.arange(n) / SR
    pop = (np.sin(2 * np.pi * (600 + 500 * np.exp(-tt * 40)) * tt) * np.exp(-tt * 35)).astype(np.float32)
    return {"whoosh": (whoosh, 0.3), "hit": (hit, 0.0), "pop": (pop, 0.0)}


def build_events(cuts, peaks, cut_sfx, peak_sfx, gap):
    ev = []
    if cut_sfx != "Off":
        ev += [(t, "whoosh") for t in cuts]
    if peak_sfx != "Off":
        ev += [(t, "hit" if peak_sfx.startswith("Impact") else "pop") for t in peaks]
    ev.sort()
    out, last = [], -99.0
    for t, k in ev:
        if t - last >= gap:
            out.append((t, k)); last = t
    return out


def write_sfx(path, dur, events, vol):
    lib = make_sfx()
    total = int(dur * SR)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        for c0 in range(0, total, SR * 60):
            c1 = min(total, c0 + SR * 60)
            buf = np.zeros(c1 - c0, np.float32)
            for t, k in events:
                snd, lead = lib[k]
                s0 = int((t - lead) * SR); s1 = s0 + len(snd)
                if s1 <= c0 or s0 >= c1:
                    continue
                a, b = max(s0, c0), min(s1, c1)
                buf[a - c0:b - c0] += snd[a - s0:b - s0]
            w.writeframes((np.clip(buf * 0.9 * vol, -1, 1) * 32767).astype("<i2").tobytes())


# ---------------- converting ----------------
def do_chunk(src, start, length, W, H, fps, fit, params, out, cb, crf=23, zoom_fn=None):
    part = out.replace(".mp4", ".part.mp4")
    rd = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{length:.2f}", "-i", src,
         "-vf", f"fps={fps},{vf_for(W, H, fit)}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    wr = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
         "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p", part],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fs, n = W * H * 3, 0
    while True:
        buf = rd.stdout.read(fs)
        if len(buf) < fs:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        if zoom_fn:
            fr = zoom_frame(fr, zoom_fn(start + n / fps))
        wr.stdin.write(cartoon(fr, *params).tobytes())
        n += 1
        if n % 20 == 0:
            cb(n)
    rd.stdout.close(); wr.stdin.close(); rd.wait(); wr.wait()
    if n == 0:
        if os.path.exists(part): os.remove(part)
        return
    if wr.returncode:
        raise RuntimeError("Video writer failed.")
    os.replace(part, out)


# ---------------- UI ----------------
st.title("Cartoon Video")
st.caption("Turns your own video into a cartoon look. Use only videos you have the right to edit.")

up = st.file_uploader("Video", type=["mp4", "mov", "mkv", "webm", "avi"])
if not up:
    st.stop()

if st.session_state.get("upkey") != (up.name, up.size):
    st.session_state.upkey = (up.name, up.size)
    st.session_state.uphash = hashlib.md5(up.getvalue()).hexdigest()[:12]
    st.session_state.pop("pv", None)
wd = os.path.join(TMP, "cartoon_" + st.session_state.uphash)
for d in glob.glob(os.path.join(TMP, "cartoon_*")):
    if d != wd:
        shutil.rmtree(d, ignore_errors=True)
os.makedirs(wd, exist_ok=True)
src = os.path.join(wd, "src" + os.path.splitext(up.name)[1].lower())
if not os.path.exists(src):
    open(src, "wb").write(up.getvalue())

ow, oh, ofps, dur = probe(src)
audio_on = has_audio(src)
st.write(f"Video: {ow}x{oh}, {ofps:.0f} fps, {dur/60:.1f} min" + ("" if audio_on else " (no audio)"))
if dur > MAX_MIN * 60:
    st.error(f"Video is longer than {MAX_MIN} minutes."); st.stop()

st.subheader("Output")
q = st.radio("Quality", [480, 720, 1080], index=1, horizontal=True, format_func=lambda x: f"{x}p")
rname = st.selectbox("Ratio", list(RATIOS))
fit = st.radio("If ratio is different", FITS, horizontal=True)
fps = st.radio("Frames per second", [15, 24, 30], index=1, horizontal=True)
maxq = st.checkbox("Max quality (sharper, bigger file, a bit slower)", True)
crf = 16 if maxq else 23
ratio = RATIOS[rname] or (ow, oh)
W, H = dims(ratio, q)

st.subheader("Cartoon look")
line = st.slider("Outline thickness", 1, 3, 2)
smooth = st.slider("Colour smoothness", 1, 3, 2)
levels = st.slider("Colour levels (fewer = more cartoon)", 4, 16, 8)
params = (line, smooth, levels)

st.subheader("Auto effects (optional)")
zmode = st.selectbox("Zoom", ZOOMS)
zamt = st.slider("Zoom strength (%)", 2, 15, 6) if zmode != "Off" else 0
cut_sfx = st.selectbox("Sound on scene changes", ["Off", "Whoosh"])
peak_sfx = st.selectbox("Sound on loud moments", ["Off", "Impact", "Pop"],
                        disabled=not audio_on)
sfx_vol = st.slider("Sound effect volume (%)", 10, 100, 35) if (cut_sfx != "Off" or peak_sfx != "Off") else 0
sfx_gap = st.slider("Minimum gap between sound effects (sec)", 0.5, 6.0, 2.0, 0.5) if sfx_vol else 2.0
sens_name = st.selectbox("Scene change sensitivity", list(SENS), index=1)
need_cuts = zmode.startswith(("Slow", "Punch")) or cut_sfx != "Off"
need_peaks = audio_on and (zmode.startswith("Pulse") or peak_sfx != "Off")
sens_key = sens_name.split()[0].lower()

if (need_cuts or need_peaks) and st.button("Analyse video (find scene changes / loud moments)"):
    with st.spinner("Analysing... this can take a few minutes for long videos"):
        c = get_cuts(src, wd, SENS[sens_name], sens_key) if need_cuts else []
        p = get_peaks(src, wd) if need_peaks else []
    st.success(f"Found {len(c)} scene changes and {len(p)} loud moments.")

st.subheader("Preview")
t_prev = st.slider("Preview at (seconds)", 0.0, max(dur - 1, 0.1), 0.0)
if st.button("Preview one frame"):
    fr = grab_frame(src, t_prev, W, H, fit)
    t0 = time.time(); out = cartoon(fr, *params); dt = time.time() - t0
    st.session_state.pv = (fr, out, dt)
if "pv" in st.session_state:
    fr, out, dt = st.session_state.pv
    c1, c2 = st.columns(2)
    c1.image(fr[:, :, ::-1], caption="Original")
    c2.image(out[:, :, ::-1], caption="Cartoon")
    est = dur * fps * dt * 1.5 / 60
    st.info(f"Rough time estimate on this server: about {est:.0f} min "
            f"({dur*fps:.0f} frames). The real time is known only after a real run.")

st.subheader("Convert")
st.caption("Keep this page open while converting. If the connection drops, upload the same "
           "video again with the same settings and finished parts will be reused.")
zkey = f"{zmode}-{zamt}-{sens_key if zmode.startswith(('Slow', 'Punch')) else ''}"
sig = hashlib.md5(f"{W}x{H}-{fps}-{fit}-{params}-{crf}-{zkey}".encode()).hexdigest()[:8]
if st.button("Convert video"):
    try:
        t0 = time.time()
        with st.spinner("Analysing video first (only if effects are on)..."):
            cuts = get_cuts(src, wd, SENS[sens_name], sens_key) if need_cuts else []
            peaks = get_peaks(src, wd) if need_peaks else []
        zoom_fn = make_zoom(zmode, zamt / 100, cuts, peaks, dur)
        n_ch = math.ceil(dur / CHUNK)
        bar = st.progress(0.0, "Starting...")
        parts = []
        for i in range(n_ch):
            start, length = i * CHUNK, min(CHUNK, dur - i * CHUNK)
            outc = os.path.join(wd, f"c_{sig}_{i:04d}.mp4")
            if not os.path.exists(outc):
                def cb(n, i=i, length=length):
                    frac = (i + min(n / (length * fps), 1)) / n_ch
                    bar.progress(min(frac, 1.0),
                                 f"Part {i+1}/{n_ch} | {(time.time()-t0)/60:.1f} min elapsed")
                do_chunk(src, start, length, W, H, fps, fit, params, outc, cb, crf, zoom_fn)
            if os.path.exists(outc):
                parts.append(outc)
            bar.progress((i + 1) / n_ch, f"Part {i+1}/{n_ch} done | {(time.time()-t0)/60:.1f} min elapsed")
        lst = os.path.join(wd, "list.txt")
        open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
        final = os.path.join(wd, f"cartoon_{sig}.mp4")
        bar.progress(1.0, "Joining parts and adding audio...")
        events = build_events(cuts, peaks, cut_sfx, peak_sfx, sfx_gap)
        base = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst]
        if events:
            sfx = os.path.join(wd, "sfx.wav")
            write_sfx(sfx, dur, events, sfx_vol / 100)
            if audio_on:
                fc = ("[1:a:0]aformat=sample_rates=44100:channel_layouts=stereo[a1];"
                      "[2:a:0]aformat=sample_rates=44100:channel_layouts=stereo[a2];"
                      "[a1][a2]amix=inputs=2:duration=first:normalize=0[a]")
                cmd = base + ["-i", src, "-i", sfx, "-filter_complex", fc, "-map", "0:v:0",
                              "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                              "-shortest", final]
            else:
                cmd = base + ["-i", sfx, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                              "-c:a", "aac", "-b:a", "160k", "-shortest", final]
        else:
            cmd = base + ["-i", src, "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
                          "-c:a", "aac", "-b:a", "128k", "-shortest", final]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-600:])
        st.session_state.cartoon_out = final
        bar.empty()
        st.success(f"Done in {(time.time()-t0)/60:.1f} min. "
                   f"Zoom: {zmode}. Sound effects added: {len(events)}.")
    except Exception as e:
        st.error(f"Error: {e}")

final = st.session_state.get("cartoon_out")
if final and os.path.exists(final):
    mb = os.path.getsize(final) / 1e6
    if mb < 80:
        st.video(final)
    else:
        st.write(f"Video is {mb:.0f} MB (preview skipped). Download it below.")
    with open(final, "rb") as f:
        st.download_button("Download MP4", f, "cartoon.mp4", "video/mp4")

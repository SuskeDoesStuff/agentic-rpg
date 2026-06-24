"""A watch-only web driver: stream a live all-agent game to the browser.

The engine is already a generator of display events; this serves a small page that
lets a visitor build a party (1-4 agents, each free-text class and personality),
then streams the resulting game over Server-Sent Events. Because no human plays,
there is nothing to send back: the request is a one-way stream, so there is no
session to hold between requests. A semaphore caps concurrent games and the round
count is clamped, so a shared link cannot run away with cost.

Run with ``rpg-web`` (or ``python -m rpg.web``); set ``OPENAI_API_KEY`` for live
games. With no key, the stream returns a single, honest error rather than faking a
game. Tracing, if configured, is handled inside ``engine.play`` per session.
"""
from __future__ import annotations

import json
import os
import threading

import networkx as nx
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import config, engine, events, players
from .world import START, WORLD, exits

ROUND_CAP = 60
MAX_CONCURRENT = 3
_slots = threading.Semaphore(MAX_CONCURRENT)

app = FastAPI(title="Agentic RPG")


def _room_layout():
    """Lay the world's rooms out once as a 2D graph the page can draw; coordinates normalized to [0,1]."""
    rooms = list(WORLD["rooms"])
    rg = nx.Graph()
    rg.add_nodes_from(rooms)
    edges, seen = [], set()
    for r in rooms:
        for nb in exits(r):
            key = frozenset((r, nb))
            if nb in rooms and key not in seen:
                seen.add(key)
                rg.add_edge(r, nb)
                edges.append([r, nb])
    try:
        pos = nx.spring_layout(rg, seed=7, iterations=300)
    except Exception:
        pos = nx.circular_layout(rg)
    xs = [p[0] for p in pos.values()] or [0.0]
    ys = [p[1] for p in pos.values()] or [0.0]
    lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)

    def norm(v, lo, hi):
        return 0.5 if hi == lo else (v - lo) / (hi - lo)

    nodes = [{"id": r, "x": round(norm(pos[r][0], lo_x, hi_x), 4),
              "y": round(norm(pos[r][1], lo_y, hi_y), 4)} for r in rooms]
    return {"nodes": nodes, "edges": edges, "start": START}


LAYOUT = _room_layout()


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


def _party_view(p):
    return {"name": p["name"], "class_name": p.get("class_name", p.get("class_desc", "")),
            "hp": p["hp"], "max_hp": p["max_hp"], "mana": p["mana"], "max_mana": p["max_mana"]}


def _serialize(ev):
    """Map an engine event to a small JSON payload, or None for events a watcher doesn't need."""
    if isinstance(ev, events.Narration):
        return {"type": "narration", "text": ev.text}
    if isinstance(ev, events.Dialogue):
        return {"type": "dialogue", "speaker": ev.speaker, "text": ev.text}
    if isinstance(ev, events.System):
        return {"type": "system", "text": ev.text}
    if isinstance(ev, events.Argument):
        return {"type": "argument", "speaker": ev.speaker, "destination": ev.destination, "reason": ev.reason}
    if isinstance(ev, events.QuestUpdate):
        return {"type": "quest", "title": ev.title, "status": ev.status}
    if isinstance(ev, events.GameOver):
        return {"type": "gameover", "won": ev.won, "reason": ev.reason}
    return None  # input-request events never arise in an all-agent game


def _build_party(spec):
    party = []
    for a in spec[:4]:
        name = (str(a.get("name", "")).strip() or "Adventurer")[:24]
        cls = (str(a.get("class_desc", "")).strip() or "a wandering adventurer")[:120]
        per = str(a.get("personality", "")).strip()[:120]
        party.append(players.make_player(name, cls, per, is_agent=True))
    return party


def _stream(spec, rounds):
    """The SSE body: build the party, then relay engine events plus room and party-state changes."""
    if not _slots.acquire(blocking=False):
        yield _sse({"type": "error", "message": "The hall is full right now. Try again in a moment."})
        return
    try:
        if not config.has_key():
            yield _sse({"type": "error",
                        "message": "No API key is configured on this server, so a live game can't run."})
            return
        if not spec:
            yield _sse({"type": "error", "message": "Add at least one agent to the party."})
            return
        try:
            party = _build_party(spec)
            gs = players.new_game(party)
        except Exception:
            yield _sse({"type": "error", "message": "That party couldn't be assembled."})
            return
        yield _sse({"type": "start", "location": gs.location, "party": [_party_view(p) for p in party]})
        last_room, last_snap = None, None
        for ev in engine.play(gs, max_rounds=rounds):
            payload = _serialize(ev)
            if payload:
                yield _sse(payload)
            if gs.location != last_room:
                last_room = gs.location
                yield _sse({"type": "location", "room": gs.location})
            snap = tuple((p["hp"], p["mana"]) for p in gs.party)
            if snap != last_snap:
                last_snap = snap
                yield _sse({"type": "party", "party": [_party_view(p) for p in gs.party]})
            if isinstance(ev, events.GameOver):
                break
    finally:
        _slots.release()


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/map")
def world_map():
    return JSONResponse(LAYOUT)


@app.get("/play")
def play(party: str = Query("[]"), rounds: int = Query(48)):
    try:
        spec = json.loads(party)
    except Exception:
        spec = []
    if not isinstance(spec, list):
        spec = []
    rounds = max(1, min(int(rounds or 48), ROUND_CAP))
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    return StreamingResponse(_stream(spec, rounds), media_type="text/event-stream", headers=headers)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Agentic RPG — a watched game</title>
<style>
  :root{
    --ink:#14181f; --ink2:#0f131a; --panel:#1b212b; --line:#2b333f;
    --parch:#e8dfce; --muted:#8a93a3; --gold:#d9a441; --steel:#8fb0cf;
    --ember:#cf6a5e; --sage:#7fb58e;
    --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{background:radial-gradient(1200px 800px at 70% -10%,#1c2430 0%,var(--ink) 55%);
       color:var(--parch);font-family:var(--serif);line-height:1.5;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1180px;margin:0 auto;padding:28px 20px 64px}
  header.top{display:flex;align-items:baseline;justify-content:space-between;gap:16px;
       border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:22px}
  .brand{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
  .brand h1{font-size:26px;font-weight:600;letter-spacing:.3px;margin:0}
  .brand .tag{font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:.14em;text-transform:uppercase}
  .ghost{font-family:var(--mono);font-size:12px;color:var(--muted);background:transparent;border:1px solid var(--line);
       border-radius:999px;padding:7px 14px;cursor:pointer;letter-spacing:.06em}
  .ghost:hover{color:var(--parch);border-color:var(--steel)}
  .ghost[hidden]{display:none}

  /* builder */
  .builder{max-width:680px;margin:6vh auto 0}
  .builder .lede{color:var(--muted);font-size:17px;margin:0 0 26px;max-width:52ch}
  .builder .lede b{color:var(--parch);font-weight:600}
  .countrow{display:flex;align-items:center;gap:10px;font-family:var(--mono);font-size:12px;
       color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:18px}
  .chip{font-family:var(--mono);font-size:14px;color:var(--muted);background:var(--panel);
       border:1px solid var(--line);border-radius:8px;width:38px;height:34px;cursor:pointer}
  .chip.on{color:var(--ink);background:var(--gold);border-color:var(--gold);font-weight:700}
  .agent{border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:12px;background:rgba(27,33,43,.5)}
  .agent .n{display:flex;align-items:center;gap:10px;margin-bottom:10px}
  .agent .seal{font-family:var(--mono);font-size:11px;color:var(--gold);border:1px solid var(--line);
       border-radius:6px;padding:3px 8px;letter-spacing:.1em}
  label.f{display:block;font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:.14em;
       text-transform:uppercase;margin:10px 0 5px}
  input,textarea{width:100%;background:var(--ink2);border:1px solid var(--line);border-radius:8px;
       color:var(--parch);font-family:var(--serif);font-size:15px;padding:9px 11px;resize:vertical}
  input:focus,textarea:focus{outline:none;border-color:var(--steel)}
  .two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .begin{margin-top:18px;width:100%;font-family:var(--mono);font-size:13px;letter-spacing:.12em;text-transform:uppercase;
       color:var(--ink);background:var(--gold);border:0;border-radius:10px;padding:14px;cursor:pointer;font-weight:700}
  .begin:hover{filter:brightness(1.06)}
  .note{font-family:var(--mono);font-size:11px;color:var(--ember);margin-top:12px;min-height:14px}

  /* game */
  .game{display:none;grid-template-columns:1fr 360px;gap:22px;align-items:start}
  .game.live{display:grid}
  .log{min-height:60vh}
  .line{padding:7px 0;border-bottom:1px solid rgba(43,51,63,.5);animation:rise .35s ease both}
  @keyframes rise{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
  .narr{font-size:17px;color:var(--parch)}
  .dlg{font-size:16px}
  .dlg .who{font-family:var(--mono);font-size:12px;letter-spacing:.04em;margin-right:8px}
  .sys,.arg,.quest{font-family:var(--mono);font-size:13px;letter-spacing:.01em}
  .sys{color:var(--steel)}
  .arg{color:var(--muted)} .arg .d{color:var(--steel)}
  .quest{display:inline-flex;align-items:center;gap:8px;color:var(--gold);border:1px solid rgba(217,164,65,.4);
       border-radius:999px;padding:5px 12px;background:rgba(217,164,65,.07);margin:3px 0}
  .quest.done{color:var(--sage);border-color:rgba(127,181,142,.4);background:rgba(127,181,142,.07)}
  .quest .k{font-size:10px;letter-spacing:.16em;text-transform:uppercase;opacity:.8}
  .over{margin-top:18px;font-family:var(--mono);font-size:14px;letter-spacing:.06em;padding:14px 16px;border-radius:10px;border:1px solid var(--line)}
  .over.won{color:var(--sage);border-color:rgba(127,181,142,.5)} .over.lost{color:var(--ember);border-color:rgba(207,106,94,.5)}

  aside{position:sticky;top:18px;display:flex;flex-direction:column;gap:18px}
  .card{background:var(--ink2);border:1px solid var(--line);border-radius:14px;padding:14px}
  .card h2{margin:0 0 10px;font-family:var(--mono);font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);font-weight:600}
  svg{width:100%;height:240px;display:block}
  .edge{stroke:#2b333f;stroke-width:1.5}
  .node{fill:#222c38;stroke:#37424f;stroke-width:1.5;transition:fill .4s,stroke .4s,r .3s}
  .node.seen{fill:#33414f}
  .node.here{fill:var(--gold);stroke:#f0c97a}
  .here-glow{fill:var(--gold);opacity:.18;animation:pulse 2.2s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:.10;r:10}50%{opacity:.26;r:16}}
  .rlabel{fill:var(--muted);font-family:var(--mono);font-size:9px;letter-spacing:.04em}
  .rlabel.here{fill:var(--gold)}
  .who-roster{display:flex;flex-direction:column;gap:12px}
  .pc .name{display:flex;justify-content:space-between;align-items:baseline}
  .pc .name b{font-weight:600;font-size:15px} .pc .name span{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:.04em}
  .bar{height:6px;border-radius:4px;background:#222c38;margin-top:6px;overflow:hidden}
  .bar > i{display:block;height:100%;border-radius:4px;background:var(--ember);transition:width .5s}
  .bar.mana{margin-top:4px} .bar.mana > i{background:var(--steel)}
  .pc.down .name b{color:var(--muted);text-decoration:line-through}
  .legend{font-family:var(--mono);font-size:10px;color:var(--muted);display:flex;gap:14px;margin-top:8px;letter-spacing:.04em}
  .legend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:middle}

  @media (max-width:860px){ .game.live{grid-template-columns:1fr} aside{position:static} .two{grid-template-columns:1fr} }
  @media (prefers-reduced-motion:reduce){ .line{animation:none} .here-glow{animation:none} }
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div class="brand">
      <h1>A Watched Game</h1>
      <span class="tag">harnessed agents · the world is a graph</span>
    </div>
    <button class="ghost" id="again" hidden>Watch another</button>
  </header>

  <!-- BUILDER -->
  <section class="builder" id="builder">
    <p class="lede">Assemble a party of <b>autonomous agents</b> and watch them reason through a small world on their own:
       negotiating where to go, fighting what guards the way, and finishing quests. You set who they are; the world holds the rules.</p>
    <div class="countrow">Party size
      <button class="chip" data-n="1">1</button>
      <button class="chip" data-n="2">2</button>
      <button class="chip" data-n="3">3</button>
      <button class="chip" data-n="4">4</button>
    </div>
    <div id="agents"></div>
    <button class="begin" id="begin">Begin the watch</button>
    <div class="note" id="note"></div>
  </section>

  <!-- GAME -->
  <section class="game" id="game">
    <div class="log" id="log"></div>
    <aside>
      <div class="card">
        <h2>The world</h2>
        <svg id="map" viewBox="0 0 100 64" preserveAspectRatio="xMidYMid meet"></svg>
        <div class="legend"><span><i style="background:var(--gold)"></i>here</span>
          <span><i style="background:#33414f"></i>seen</span>
          <span><i style="background:#222c38"></i>unseen</span></div>
      </div>
      <div class="card">
        <h2>The party</h2>
        <div class="who-roster" id="roster"></div>
      </div>
    </aside>
  </section>
</div>

<script>
const PRESETS = [
  {name:"Borin", class_desc:"a katana-wielding vanguard, all forward pressure", personality:"bold and wry"},
  {name:"Sable", class_desc:"a battle-mage who leads with fire", personality:"sharp, loyal, calm under fire"},
  {name:"Wren",  class_desc:"a quick scout who fights with a spear", personality:"cautious, watchful"},
  {name:"Cael",  class_desc:"a steadfast cleric who mends wounds", personality:"gentle, stubborn in a pinch"},
];
const SPEAKER_COLORS = ["#e6c07b","#9ec1e0","#c3a6d8","#a8cfa0","#e3a17c","#7fc6c6","#d7a0a0","#9fb8d0"];
function colorFor(name){ let h=0; for(const c of (name||"")) h=(h*31+c.charCodeAt(0))>>>0; return SPEAKER_COLORS[h%SPEAKER_COLORS.length]; }

let count = 2;
const agentsEl = document.getElementById("agents");
function renderAgents(){
  agentsEl.innerHTML = "";
  for(let i=0;i<count;i++){
    const p = PRESETS[i] || {name:"",class_desc:"",personality:""};
    const d = document.createElement("div"); d.className="agent";
    d.innerHTML = `<div class="n"><span class="seal">Agent ${i+1}</span></div>
      <label class="f">Name</label><input data-k="name" value="${esc(p.name)}" maxlength="24"/>
      <div class="two">
        <div><label class="f">Class</label><input data-k="class_desc" value="${esc(p.class_desc)}" maxlength="120"/></div>
        <div><label class="f">Personality</label><input data-k="personality" value="${esc(p.personality)}" maxlength="120"/></div>
      </div>`;
    agentsEl.appendChild(d);
  }
}
function esc(s){ return (s||"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c])); }
document.querySelectorAll(".chip").forEach(c=>c.addEventListener("click",()=>{
  count = +c.dataset.n;
  document.querySelectorAll(".chip").forEach(x=>x.classList.toggle("on", x===c));
  renderAgents();
}));
document.querySelector('.chip[data-n="2"]').classList.add("on");
renderAgents();

function gatherParty(){
  const out=[];
  document.querySelectorAll(".agent").forEach(a=>{
    const o={}; a.querySelectorAll("input").forEach(i=>o[i.dataset.k]=i.value.trim());
    if(o.name && o.class_desc) out.push(o);
  });
  return out;
}

// ---- map ----
let LAYOUT=null, nodeEls={}, labelEls={}, glowEl=null, seen=new Set();
fetch("/map").then(r=>r.json()).then(d=>LAYOUT=d).catch(()=>{});
const SVG="http://www.w3.org/2000/svg";
function drawMap(){
  const map=document.getElementById("map"); map.innerHTML=""; nodeEls={}; labelEls={}; seen=new Set();
  if(!LAYOUT) return;
  const X=v=>6+v*88, Y=v=>8+v*48;
  for(const [a,b] of LAYOUT.edges){
    const na=LAYOUT.nodes.find(n=>n.id===a), nb=LAYOUT.nodes.find(n=>n.id===b);
    const l=document.createElementNS(SVG,"line");
    l.setAttribute("x1",X(na.x));l.setAttribute("y1",Y(na.y));
    l.setAttribute("x2",X(nb.x));l.setAttribute("y2",Y(nb.y));
    l.setAttribute("class","edge"); map.appendChild(l);
  }
  glowEl=document.createElementNS(SVG,"circle"); glowEl.setAttribute("class","here-glow");
  glowEl.setAttribute("r","12"); glowEl.style.display="none"; map.appendChild(glowEl);
  for(const n of LAYOUT.nodes){
    const c=document.createElementNS(SVG,"circle");
    c.setAttribute("cx",X(n.x));c.setAttribute("cy",Y(n.y));c.setAttribute("r","4.5");
    c.setAttribute("class","node"); map.appendChild(c); nodeEls[n.id]=c;
    const t=document.createElementNS(SVG,"text");
    t.setAttribute("x",X(n.x));t.setAttribute("y",Y(n.y)-7);t.setAttribute("text-anchor","middle");
    t.setAttribute("class","rlabel"); t.textContent=n.id; map.appendChild(t); labelEls[n.id]=t;
  }
}
function setHere(room){
  if(!LAYOUT||!nodeEls[room]) return;
  Object.entries(nodeEls).forEach(([id,el])=>{
    el.classList.remove("here"); if(seen.has(id)) el.classList.add("seen");
    labelEls[id].classList.remove("here");
  });
  seen.add(room);
  nodeEls[room].classList.add("here"); nodeEls[room].classList.remove("seen");
  labelEls[room].classList.add("here");
  const n=LAYOUT.nodes.find(x=>x.id===room);
  glowEl.style.display=""; glowEl.setAttribute("cx",6+n.x*88); glowEl.setAttribute("cy",8+n.y*48);
}

// ---- roster ----
function renderRoster(party){
  const r=document.getElementById("roster"); r.innerHTML="";
  party.forEach(p=>{
    const hp=Math.max(0,Math.round(100*p.hp/Math.max(1,p.max_hp)));
    const mp=p.max_mana>0?Math.max(0,Math.round(100*p.mana/p.max_mana)):0;
    const d=document.createElement("div"); d.className="pc"+(p.hp<=0?" down":"");
    d.innerHTML=`<div class="name"><b style="color:${colorFor(p.name)}">${esc(p.name)}</b>
        <span>${esc(p.class_name||"")} · ${p.hp}/${p.max_hp} hp</span></div>
      <div class="bar"><i style="width:${hp}%"></i></div>
      ${p.max_mana>0?`<div class="bar mana"><i style="width:${mp}%"></i></div>`:""}`;
    r.appendChild(d);
  });
}

// ---- stream ----
const logEl=document.getElementById("log");
function add(node){ logEl.appendChild(node); window.scrollTo({top:document.body.scrollHeight,behavior:"smooth"}); }
function div(cls,html){ const d=document.createElement("div"); d.className="line "+cls; d.innerHTML=html; return d; }

let es=null, finished=false;
function handle(ev){
  switch(ev.type){
    case "start": renderRoster(ev.party); drawMap(); setHere(ev.location); break;
    case "party": renderRoster(ev.party); break;
    case "location": setHere(ev.room); break;
    case "narration": add(div("narr", esc(ev.text))); break;
    case "dialogue": add(div("dlg", `<span class="who" style="color:${colorFor(ev.speaker)}">${esc(ev.speaker)}</span>${esc(ev.text)}`)); break;
    case "system": add(div("sys", esc(ev.text))); break;
    case "argument": add(div("arg", `${esc(ev.speaker)} argues for <span class="d">${esc(ev.destination)}</span> — ${esc(ev.reason)}`)); break;
    case "quest": {
      const done=ev.status==="completed";
      add(div("", `<span class="quest ${done?"done":""}"><span class="k">Quest ${esc(ev.status)}</span>${esc(ev.title)}</span>`)); break; }
    case "gameover": {
      finished=true; if(es) es.close();
      const won=ev.won;
      const o=document.createElement("div"); o.className="over "+(won?"won":"lost");
      o.textContent = won ? "Every quest complete. The party prevails." : "The run ends here — "+(ev.reason||"the party falls.");
      logEl.appendChild(o); document.getElementById("again").hidden=false;
      window.scrollTo({top:document.body.scrollHeight,behavior:"smooth"}); break; }
    case "error": {
      finished=true; if(es) es.close();
      logEl.appendChild(div("sys", '<span style="color:var(--ember)">'+esc(ev.message)+"</span>"));
      document.getElementById("again").hidden=false; break; }
  }
}

function begin(){
  const party=gatherParty();
  const note=document.getElementById("note");
  if(party.length===0){ note.textContent="Give each agent at least a name and a class."; return; }
  note.textContent="";
  document.getElementById("builder").style.display="none";
  document.getElementById("game").classList.add("live");
  logEl.innerHTML=""; finished=false;
  const q=encodeURIComponent(JSON.stringify(party));
  es=new EventSource("/play?rounds=48&party="+q);
  es.onmessage=e=>{ try{ handle(JSON.parse(e.data)); }catch(_){ } };
  es.onerror=()=>{ if(!finished){ /* let the browser retry transient drops, but never after we're done */ } };
}
document.getElementById("begin").addEventListener("click",begin);
document.getElementById("again").addEventListener("click",()=>{
  if(es) es.close();
  document.getElementById("again").hidden=true;
  document.getElementById("game").classList.remove("live");
  document.getElementById("builder").style.display="";
  document.getElementById("note").textContent="";
});
</script>
</body>
</html>
"""

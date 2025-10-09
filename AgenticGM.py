# agentic_gm_spoke_wheel.py
import math
import re
from typing import TypedDict, List, Dict, Annotated, Optional, Callable
from pathlib import Path
from operator import add

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# =========================
# Config & helpers
# =========================
LLM_MODEL = "XXX"          # placeholder
LORE_FILE = Path("LORE.txt")
_TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
N_TURNS = 3  # hard cap for the GM <-> Storyteller loop


class HashingEmbeddings(Embeddings):
    """Deterministic, lightweight embeddings using hashed token counts."""

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        for token in self._tokenize(text):
            idx = hash(token) % self.dimension
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


_embeddings_factory: Optional[Callable[[], Embeddings]] = None
_cached_embeddings: Optional[Embeddings] = None


def set_embeddings_factory(factory: Optional[Callable[[], Embeddings]]) -> None:
    """Configure a factory used to construct embeddings for lore retrieval."""

    global _embeddings_factory, _cached_embeddings
    _embeddings_factory = factory
    _cached_embeddings = None


def _create_default_embeddings() -> Embeddings:
    try:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings()
    except Exception:
        return HashingEmbeddings()


def _get_embeddings() -> Embeddings:
    global _cached_embeddings

    if _embeddings_factory is not None:
        return _embeddings_factory()

    if _cached_embeddings is None:
        _cached_embeddings = _create_default_embeddings()

    return _cached_embeddings

def is_risky(state) -> bool:
    q = (state.get("query") or "").lower()
    um = (state.get("user_message") or "").lower()
    return "risky" in q or "risky" in um

# =========================
# State
# =========================
class GameState(TypedDict):
    user_message: str
    query: str

    # core narrative signals
    lore: str
    history: Annotated[List[str], add]
    story: str

    # dice & inventory
    dice_roll: int
    inventory: Dict[str, int]

    # GM loop control
    turn_count: int                      # increments each time GM processes a story beat
    require_qm: bool                     # GM requests a state update (e.g., item used)
    inventory_delta: Optional[Dict[str, int]]  # {item: +/−amount} to apply, then clear
    gm_should_increment: bool            # storyteller signals whether the next GM pass advances the turn

    # final/aux output
    output: str

# =========================
# Tools
# =========================
class RandomizerTool:
    def roll(self, sides=20):
        import random
        return random.randint(1, sides)

randomizer = RandomizerTool()

# =========================
# Agents (nodes)
# =========================
def orchestrator(state: GameState):
    # No persistent state change here; routing is handled by conditional edges.
    return {}

def gambler(state: GameState):
    return {"dice_roll": randomizer.roll(20)}

def historian(state: GameState):
    """Ephemeral RAG over LORE.txt (no persistent vector DB)."""
    if not LORE_FILE.exists():
        return {"lore": ""}

    raw = LORE_FILE.read_text(encoding="utf-8").strip()
    if not raw:
        return {"lore": ""}

    chunks = [c.strip() for c in _TEXT_SPLITTER.split_text(raw) if c.strip()]
    if not chunks:
        return {"lore": ""}

    embeddings = _get_embeddings()
    vs = FAISS.from_texts(chunks, embeddings)

    query = (state.get("query") or "").strip()
    if not query:
        selected = chunks[:1]
    else:
        docs = vs.similarity_search(query, k=min(3, len(chunks)))
        selected = [d.page_content for d in docs] or chunks[:1]

    return {"lore": "\n\n".join(selected)}

def journal(state: GameState):
    msg = state.get("user_message", "")
    return {"history": [msg] if msg else []}

def storyteller(state: GameState):
    lore = state.get("lore", "")
    roll = state.get("dice_roll", 0)
    inv = state.get("inventory", {})
    # Stub generation (keep it deterministic for now)
    text = f"[LLM:{LLM_MODEL}] Story beat (turn={state.get('turn_count',0)}): roll={roll}, inv={inv}\nLore used: {bool(lore)}"
    return {"story": text, "gm_should_increment": True}

def gm(state: GameState):
    """
    - Increments turn_count
    - Optionally sets require_qm and inventory_delta (e.g., adjust inventory counts)
    - Decides nothing here; routing occurs via conditional edges after GM
    """
    should_increment = bool(state.get("gm_should_increment", False))
    current_turn = int(state.get("turn_count", 0))
    turn = current_turn + 1 if should_increment else current_turn

    # Example: if the narrative references an inventory item, request a decrement
    # For demo: when dice_roll <= 5 on first loop, pretend the party consumes the first item
    require_qm = False
    delta = None
    def _usable_inventory_items() -> List[str]:
        inventory = state.get("inventory") or {}
        return [item for item, count in inventory.items() if count > 0]

    def _item_from_query(items: List[str]) -> Optional[str]:
        query = (state.get("query") or "").lower()
        if not query:
            return None

        for item in items:
            if item.lower() in query:
                return item
        return None

    if should_increment:
        usable_items = _usable_inventory_items()
        selected_item = _item_from_query(usable_items)

        if not selected_item and turn == 1 and state.get("dice_roll", 0) <= 5:
            # illustrative narrative side-effect
            selected_item = usable_items[0] if usable_items else None

        if selected_item:
            require_qm = True
            delta = {selected_item: -1}

    out = (
        f"GM Narration (turn={turn}):\n"
        f"{state.get('story','')}\n"
        f"GM requests update? {require_qm}; delta={delta}"
    )
    return {
        "turn_count": turn,
        "require_qm": require_qm,
        "inventory_delta": delta,
        "output": out,
        "gm_should_increment": False,
    }

def quartermaster(state: GameState):
    """
    Applies inventory_delta (if any), then clears it.
    Safety: routing is enforced by conditional edges (QM → Storyteller ONLY if turn_count == 1).
    """
    inv = dict(state.get("inventory", {}))
    delta = state.get("inventory_delta") or {}
    for k, v in delta.items():
        inv[k] = inv.get(k, 0) + v
        if inv[k] <= 0:
            inv.pop(k, None)
    # Clear the delta after applying
    return {"inventory": inv, "inventory_delta": None, "require_qm": False, "gm_should_increment": False}

# =========================
# Routing functions (for conditional edges)
# =========================
def route_from_orchestrator(state: GameState) -> str:
    return "gambler" if is_risky(state) else "historian"

def after_gm(state: GameState) -> str:
    """
    Rules:
      - If turn_count >= N_TURNS:
          - if require_qm: go to quartermaster, then (QM → GM) and GM → END
          - else: END directly
      - If turn_count < N_TURNS:
          - if require_qm: quartermaster
          - else: storyteller
    """
    turn = int(state.get("turn_count", 0))
    need_qm = bool(state.get("require_qm", False))

    if turn >= N_TURNS:
        return "quartermaster" if need_qm else "end"

    # turn < N_TURNS
    if need_qm:
        return "quartermaster"
    return "storyteller"

def after_quartermaster(state: GameState) -> str:
    """
    Safety rule:
      - If turn_count == 1  -> storyteller (allow continuing the loop)
      - If turn_count >= 2  -> gm (NEVER back to storyteller)
    """
    turn = int(state.get("turn_count", 0))
    return "storyteller" if turn == 1 else "gm"

# =========================
# Build Graph
# =========================
graph = StateGraph(GameState)

# Register nodes
graph.add_node("orchestrator", orchestrator)
graph.add_node("gambler", gambler)
graph.add_node("historian", historian)
graph.add_node("journal", journal)
graph.add_node("storyteller", storyteller)
graph.add_node("gm", gm)
graph.add_node("quartermaster", quartermaster)

# Entry
graph.set_entry_point("orchestrator")

# Conditionals and edges
graph.add_conditional_edges(
    "orchestrator",
    route_from_orchestrator,
    {"gambler": "gambler", "historian": "historian"}
)

graph.add_edge("gambler", "storyteller")
graph.add_edge("historian", "journal")
graph.add_edge("journal", "storyteller")
graph.add_edge("storyteller", "gm")

graph.add_conditional_edges(
    "gm",
    after_gm,
    {"storyteller": "storyteller", "quartermaster": "quartermaster", "end": END}
)

graph.add_conditional_edges(
    "quartermaster",
    after_quartermaster,
    {"storyteller": "storyteller", "gm": "gm"}
)

# Compile
app = graph.compile()

# Optional: render a PNG (works when Graphviz is available)
try:
    graph.draw_png("agentic_wheel.png")
    print("Wrote agentic_wheel.png")
except Exception as e:
    print("Diagram export skipped:", e)

# --- Example run (feel free to comment out) ---
if __name__ == "__main__":
    initial_state: GameState = {
        "user_message": "The party considers a risky shortcut through the catacombs.",
        "query": "risky shortcut, maybe use a potion",
        "lore": "",
        "history": [],
        "story": "",
        "dice_roll": 0,
        "inventory": {"potion": 1, "gold": 10},
        "turn_count": 0,
        "require_qm": False,
        "inventory_delta": None,
        "gm_should_increment": False,
        "output": ""
    }
    final = app.run(initial_state)
    print("\n=== FINAL OUTPUT ===\n", final["output"])
    print("\nInventory:", final["inventory"])
    print("Turn count:", final["turn_count"])

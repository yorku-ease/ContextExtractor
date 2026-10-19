# adapters/__init__.py

from adapters.wildchat import WildChatAdapter
from adapters.lmsys import LMSYSAdapter
from adapters.clariq import ClariQAdapter
from adapters.ambigqa import AmbigQAAdapter

ADAPTERS = {
    "allenai/WildChat": WildChatAdapter,
    "lmsys/lmsys-chat-1m": LMSYSAdapter,
    "convai-challenge/conv_ai_3": ClariQAdapter,
    "sewon/ambig_qa": AmbigQAAdapter,
}
# adapters/__init__.py

from adapters.wildchat import WildChatAdapter
#from adapters.lmsys import LMSYSAdapter
#from adapters.clariq import ClariQAdapter
#from adapters.ambigqa import AmbigQAAdapter

ADAPTERS = {
    "allenai/WildChat": WildChatAdapter,
    #"lmsys/lmsys-chat-1m": LMSYSAdapter,
    #"clariq": ClariQAdapter,
    #"ambigqa": AmbigQAAdapter,
}
from .models import STTRealtimeSampleRates, TTSEncoding, TTSModels
from .stt import STT, SpeechStream
from .tts import DEFAULT_VOICE_ID, TTS, Voice, VoiceSettings
from .version import __version__

__all__ = [
    "STT",
    "SpeechStream",
    "TTS",
    "Voice",
    "VoiceSettings",
    "TTSEncoding",
    "TTSModels",
    "STTRealtimeSampleRates",
    "DEFAULT_VOICE_ID",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class BlazePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(BlazePlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False

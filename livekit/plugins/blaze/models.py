from typing import Literal

TTSModels = Literal[
    "1.5-realtime",
    "2.0-realtime",
    "2.0-beta"
]

TTSEncoding = Literal[
    "wav",
    "mp3",
    "pcm",
    "opus",
]

STTRealtimeSampleRates = Literal[
    16000,
]

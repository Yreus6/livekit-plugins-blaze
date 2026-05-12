from __future__ import annotations

import asyncio
import json
import os
import time
import weakref
from dataclasses import dataclass, replace
from typing import Any, Literal

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    tokenize,
    tts,
    utils,
)
from livekit.agents.tts import ChunkedStream
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import TTSEncoding, TTSModels

_DefaultEncoding: TTSEncoding = "pcm"


def _encoding_to_mimetype(encoding: TTSEncoding) -> str:
    if encoding == "mp3":
        return "audio/mp3"
    elif encoding == "opus":
        return "audio/opus"
    elif encoding == "pcm":
        return "audio/pcm"
    elif encoding == "wav":
        return "audio/wav"
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")


@dataclass
class VoiceSettings:
    quality: NotGivenOr[int] = NOT_GIVEN  # [32, 64, 128]
    speed: NotGivenOr[float] = NOT_GIVEN  # [0.75 - 1.25]


@dataclass
class Voice:
    id: str
    name: str
    category: str


DEFAULT_VOICE_ID = "HN-Nam-2-BL"
API_BASE_URL_V1 = "https://api.blaze.vn/v1/tts"
AUTHORIZATION_HEADER = "Authorization"

_WS_CONNECTED_EVENT = "successful-connection"
_WS_AUTHENTICATED_EVENT = "successful-authentication"
_WS_PROCESSING_EVENT = "processing-request"
_WS_STREAM_STARTED_EVENT = "started-byte-stream"
_WS_STREAM_FINISHED_EVENT = "finished-byte-stream"
_WS_ERROR_EVENTS = {"failed-request", "internal-error", "bad-request"}


def _to_blaze_ws_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.startswith("https://"):
        return f"wss://{normalized[len('https://'):]}/realtime"
    if normalized.startswith("http://"):
        return f"ws://{normalized[len('http://'):]}/realtime"
    return f"{normalized}/realtime"


def _event_name(payload: dict[str, Any]) -> str:
    return str(payload.get("type") or payload.get("status") or "")


def _build_tts_request(
    *,
    text: str,
    language: str,
    encoding: TTSEncoding,
    voice_id: str,
    normalization: Literal["no", "basic", "advanced"],
    model: str,
    voice_settings: VoiceSettings,
    sample_rate: int
) -> dict[str, Any]:
    return {
        "query": text,
        "language": language,
        "audio_format": encoding,
        "audio_quality": voice_settings.quality,
        "audio_speed": voice_settings.speed,
        "speaker_id": voice_id,
        "normalization": normalization,
        "model": model,
        "sample_rate": sample_rate,
    }


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        voice_settings: NotGivenOr[VoiceSettings] = NOT_GIVEN,
        model: TTSModels | str = "1.5-realtime",
        encoding: NotGivenOr[TTSEncoding] = NOT_GIVEN,
        sample_rate: int = 24000,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        apply_text_normalization: Literal["no", "basic", "advanced"] = "basic",
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer | tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Blake TTS.

        Args:
            voice_id (str): Voice ID. Defaults to `DEFAULT_VOICE_ID`.
            voice_settings (NotGivenOr[VoiceSettings]): Voice settings.
            model (TTSModels | str): TTS model to use. Defaults to "1.5-realtime".
            api_key (NotGivenOr[str]): Blaze API key. Can be set via argument or `BLAZE_API_KEY` environment variable.
            base_url (NotGivenOr[str]): Custom base URL for the API. Optional.
            apply_text_normalization (Literal["no", "basic", "advanced"]): Controls text normalization with modes "no", "basic", and "advanced".
            word_tokenizer (NotGivenOr[tokenize.WordTokenizer | tokenize.SentenceTokenizer]): Tokenizer for processing text. Defaults to basic WordTokenizer when auto_mode=False, `livekit.agents.tokenize.blingfire.SentenceTokenizer` otherwise.
            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests. Optional.
            language (NotGivenOr[str]): Language code for the TTS model
        """

        if not is_given(encoding):
            encoding = _DefaultEncoding

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=False,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        blaze_api_key = (
            api_key if is_given(api_key) else os.environ.get("BLAZE_API_KEY")
        )
        if not blaze_api_key:
            raise ValueError(
                "Blaze API key is required, either as argument or set BLAZE_API_KEY environmental variable"
            )

        if not is_given(word_tokenizer):
            word_tokenizer = tokenize.blingfire.SentenceTokenizer()

        self._opts = _TTSOptions(
            voice_id=voice_id,
            voice_settings=voice_settings,
            model=model,
            api_key=blaze_api_key,
            base_url=base_url if is_given(base_url) else API_BASE_URL_V1,
            encoding=encoding,
            sample_rate=self.sample_rate,
            word_tokenizer=word_tokenizer,
            language=LanguageCode(language) if is_given(language) else NOT_GIVEN,
            apply_text_normalization=apply_text_normalization,
        )
        self._session = http_session
        self._streams: weakref.WeakSet[SynthesizeStream] = weakref.WeakSet()

        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=3600,
            mark_refreshed_on_get=False,
        )

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        ws_url = _to_blaze_ws_url(self._opts.base_url)
        ws = await asyncio.wait_for(
            session.ws_connect(ws_url),
            timeout,
        )

        msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
        if msg.type != aiohttp.WSMsgType.TEXT:
            raise APIConnectionError(
                f"Unexpected Blaze WS handshake message type: {msg.type}"
            )
        handshake = json.loads(msg.data)
        if _event_name(handshake) != _WS_CONNECTED_EVENT:
            raise APIConnectionError(
                f"Unexpected Blaze WS handshake response: {handshake}"
            )

        await ws.send_str(
            json.dumps({"token": self._opts.api_key, "strategy": "request"})
        )
        auth_msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
        if auth_msg.type != aiohttp.WSMsgType.TEXT:
            raise APIConnectionError(
                f"Unexpected Blaze WS auth message type: {auth_msg.type}"
            )
        auth_payload = json.loads(auth_msg.data)
        if _event_name(auth_payload) != _WS_AUTHENTICATED_EVENT:
            raise APIConnectionError(f"Blaze WS auth failed: {auth_payload}")

        logger.debug(
            "Established Blaze TTS WebSocket connection", extra={"url": ws_url}
        )
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Blaze"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def list_voices(self) -> list[Voice]:
        async with self._ensure_session().get(
            f"{self._opts.base_url}/list-speaker-ids",
            headers={AUTHORIZATION_HEADER: f"Bearer {self._opts.api_key}"},
        ) as resp:
            return _dict_to_voices_list(await resp.json())

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        voice_settings: NotGivenOr[VoiceSettings] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            voice_id (NotGivenOr[str]): Voice ID.
            voice_settings (NotGivenOr[VoiceSettings]): Voice settings.
            model (NotGivenOr[TTSModels | str]): TTS model to use.
            language (NotGivenOr[str]): Language code for the TTS model.
        """
        if is_given(model) and model != self._opts.model:
            self._opts.model = model

        if is_given(voice_id) and voice_id != self._opts.voice_id:
            self._opts.voice_id = voice_id

        if is_given(voice_settings):
            self._opts.voice_settings = voice_settings

        if is_given(language):
            language = LanguageCode(language)
            if language != self._opts.language:
                self._opts.language = language

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        raise NotImplementedError("Not supported")

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        segments_ch = utils.aio.Chan[tokenize.WordStream]()
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type=_encoding_to_mimetype(self._opts.encoding),
            stream=True,
        )

        async def _tokenize_input() -> None:
            # Converts incoming text into WordStreams and sends them into segments_ch
            word_stream = None
            async for input_item in self._input_ch:
                if isinstance(input_item, str):
                    if word_stream is None:
                        word_stream = self._opts.word_tokenizer.stream()
                        segments_ch.send_nowait(word_stream)
                    word_stream.push_text(input_item)
                elif isinstance(input_item, self._FlushSentinel):
                    if word_stream:
                        word_stream.end_input()
                    word_stream = None

            segments_ch.close()

        async def _run_segments() -> None:
            async for word_stream in segments_ch:
                await self._run_ws(word_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self, word_stream: tokenize.WordStream, output_emitter: tts.AudioEmitter
    ) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)
        input_sent_event = asyncio.Event()

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            words: list[str] = []
            async for word in word_stream:
                words.append(word.token)

            segment_text = " ".join(words).strip()
            if not segment_text:
                input_sent_event.set()
                return

            speak_msg = _build_tts_request(
                text=segment_text,
                language=self._opts.language if is_given(self._opts.language) else "vi",
                encoding=self._opts.encoding,
                voice_id=self._opts.voice_id,
                normalization=self._opts.apply_text_normalization,
                model=str(self._opts.model),
                voice_settings=self._opts.voice_settings,
                sample_rate=self._opts.sample_rate,
            )
            self._mark_started()
            await ws.send_str(json.dumps(speak_msg))
            send_task.request_start_time = time.perf_counter()
            input_sent_event.set()

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            await input_sent_event.wait()
            first_audio_logged = False

            while True:
                msg = await ws.receive()

                if msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Blaze websocket connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type == aiohttp.WSMsgType.BINARY:
                    if not first_audio_logged:
                        request_start_time = getattr(
                            send_task, "request_start_time", None
                        )
                        if request_start_time is not None:
                            ttfb_ms = (time.perf_counter() - request_start_time) * 1000
                            logger.info("Blaze TTS TTFB: %.0f ms", ttfb_ms)
                        first_audio_logged = True

                    if msg.data == b"\x00" * len(msg.data):
                        continue

                    output_emitter.push(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    resp = json.loads(msg.data)
                    mtype = _event_name(resp)
                    if mtype == _WS_STREAM_FINISHED_EVENT:
                        logger.debug("Blaze stream event: %s", resp)
                        output_emitter.end_segment()
                        break
                    elif mtype in (_WS_PROCESSING_EVENT, _WS_STREAM_STARTED_EVENT):
                        logger.debug("Blaze stream event: %s", resp)
                    elif mtype in _WS_ERROR_EVENTS:
                        raise APIStatusError(
                            message=resp.get("message", str(resp)),
                            status_code=500,
                            body=resp,
                        )
                    else:
                        logger.debug("Unknown Blaze TTS message type: %s", resp)

        async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
            self._acquire_time = self._tts._pool.last_acquire_time
            self._connection_reused = self._tts._pool.last_connection_reused
            tasks = [
                asyncio.create_task(send_task(ws)),
                asyncio.create_task(recv_task(ws)),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                input_sent_event.set()
                await utils.aio.gracefully_cancel(*tasks)


@dataclass
class _TTSOptions:
    api_key: str
    voice_id: str
    voice_settings: NotGivenOr[VoiceSettings]
    model: TTSModels | str
    language: NotGivenOr[LanguageCode]
    base_url: str
    encoding: TTSEncoding
    sample_rate: int
    word_tokenizer: tokenize.WordTokenizer | tokenize.SentenceTokenizer
    apply_text_normalization: Literal["no", "basic", "advanced"]


def _dict_to_voices_list(data: dict[str, Any]) -> list[Voice]:
    voices: list[Voice] = []
    for voice in data["list_speakers"]:
        voices.append(
            Voice(id=voice["id"], name=voice["name"], category=voice["category"])
        )

    return voices

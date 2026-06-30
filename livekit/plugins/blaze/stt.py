from __future__ import annotations

import asyncio
import json
import os
import weakref
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, http_context, is_given
from livekit.agents.voice.io import TimedString

from ._utils import PeriodicCollector
from .log import logger
from .models import STTRealtimeSampleRates

API_BASE_URL_V1 = "https://api.blaze.vn/v1/stt"
AUTHORIZATION_HEADER = "Authorization"

BlazeSTTModels = Literal["stt-async-1.0", "stt-async-1.5", "stt-stream-1.5"]


@dataclass
class STTOptions:
    model_id: BlazeSTTModels | str
    api_key: str
    base_url: str
    language_code: LanguageCode | None
    include_timestamps: bool
    enable_refinement: bool
    sample_rate: STTRealtimeSampleRates
    enable_log: bool


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        language_code: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: STTRealtimeSampleRates = 16000,
        http_session: aiohttp.ClientSession | None = None,
        model_id: NotGivenOr[BlazeSTTModels | str] = NOT_GIVEN,
        include_timestamps: bool = False,
        enable_refinement: bool = False,
        enable_log: bool = False,
    ) -> None:
        """
        Create a new instance of Blaze STT.

        Args:
            api_key (NotGivenOr[str]): Blaze API key. Can be set via argument or `BLAZE_API_KEY` environment variable.
            base_url (NotGivenOr[str]): Custom base URL for the API. Optional.
            language_code (NotGivenOr[str]): Language code for the STT model. Optional.
            sample_rate (STTRealtimeSampleRates): Audio sample rate in Hz. Default is 16000.
            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests. Optional.
            model_id (BlazeSTTModels | str): Blaze STT model to use.
        """

        model_id = model_id if is_given(model_id) else "stt-async-1.5"
        use_realtime = model_id == "stt-stream-1.5"

        super().__init__(
            capabilities=STTCapabilities(
                streaming=use_realtime,
                interim_results=True,
                aligned_transcript="word" if include_timestamps and not use_realtime else False,
            )
        )

        blaze_api_key = api_key if is_given(api_key) else os.environ.get("BLAZE_API_KEY")
        if not blaze_api_key:
            raise ValueError(
                "Blaze API key is required, either as argument or "
                "set BLAZE_API_KEY environmental variable"
            )

        self._opts = STTOptions(
            api_key=blaze_api_key,
            base_url=base_url if is_given(base_url) else API_BASE_URL_V1,
            language_code=LanguageCode(language_code) if language_code else None,
            sample_rate=sample_rate,
            include_timestamps=include_timestamps,
            enable_refinement=enable_refinement,
            model_id=model_id,
            enable_log=enable_log,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.model_id

    @property
    def provider(self) -> str:
        return "Blaze"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        if is_given(language):
            self._opts.language_code = LanguageCode(language)

        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

        params = {
            "model": self._opts.model_id,
        }

        if self._opts.language_code:
            params["language"] = self._opts.language_code

        if self._opts.include_timestamps:
            params["enable_segments"] = str(self._opts.include_timestamps).lower()

        if self._opts.enable_refinement:
            params["enable_refinement"] = str(self._opts.enable_refinement).lower()

        try:
            async with self._ensure_session().post(
                f"{self._opts.base_url}/execute",
                params=params,
                data=wav_bytes,
                headers={
                    AUTHORIZATION_HEADER: f"Bearer {self._opts.api_key}",
                    "Content-Type": "audio/wav",
                },
            ) as response:
                response_json = await response.json()
                if response.status != 200:
                    raise APIStatusError(
                        message=response_json.get("detail", "Unknown Blaze error"),
                        status_code=response.status,
                        request_id=None,
                        body=response_json,
                    )

                extracted_result = response_json.get("result")
                if extracted_result.get("status_code") != 200:
                    raise APIStatusError(
                        message=extracted_result.get("error", "Unknown Blaze error"),
                        status_code=response.status,
                        request_id=None,
                        body=response_json,
                    )

                extracted_data = extracted_result.get("data")
                extracted_text = extracted_data.get("transcription")
                speaker_id = None
                start_time, end_time = 0, 0
                words = response_json.get("segments")
                if words:
                    speaker_id = words[0].get("id", None)
                    start_time = min(w.get("startTime", 0) for w in words)
                    end_time = max(w.get("endTime", 0) for w in words)

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e

        normalized_language = LanguageCode(self._opts.language_code or "")
        return self._transcription_to_speech_event(
            language_code=normalized_language,
            text=extracted_text,
            start_time=start_time,
            end_time=end_time,
            speaker_id=speaker_id,
            words=words,
        )

    def _transcription_to_speech_event(
        self,
        language_code: str,
        text: str,
        start_time: float,
        end_time: float,
        speaker_id: str | None,
        words: list[dict[str, Any]] | None = None,
    ) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=text,
                    language=LanguageCode(language_code),
                    speaker_id=speaker_id,
                    start_time=start_time,
                    end_time=end_time,
                    words=[
                        TimedString(
                            text=word.get("text", ""),
                            start_time=word.get("startTime", 0),
                            end_time=word.get("endTime", 0),
                        )
                        for word in words
                    ]
                    if words
                    else None,
                )
            ],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        stream = SpeechStream(
            stt=self,
            opts=self._opts,
            conn_options=conn_options,
            language=language if is_given(language) else self._opts.language_code,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream


class SpeechStream(stt.SpeechStream):
    """Streaming speech recognition using Blaze realtime STT over WebSocket."""

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        language: str | None,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._language = language
        self._session = http_session
        self._reconnect_event = asyncio.Event()
        self._speaking = False  # Track if we're currently in a speech segment
        self._audio_duration_collector = PeriodicCollector(
            callback=self._on_audio_duration_report,
            duration=5.0,
        )
        self._send_task_event = asyncio.Event()

    def _session_language(self) -> str | None:
        if self._language is not None:
            return str(self._language)
        if self._opts.language_code is not None:
            return str(self._opts.language_code)
        return None

    async def _send_config(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        payload = {
            "token": self._opts.api_key,
            "language": self._opts.language_code or "vi",
            "model": self._opts.model_id,
            "enable_log": self._opts.enable_log,
        }
        await ws.send_str(json.dumps(payload))

    def _on_audio_duration_report(self, duration: float) -> None:
        usage_event = stt.SpeechEvent(
            type=stt.SpeechEventType.RECOGNITION_USAGE,
            alternatives=[],
            recognition_usage=stt.RecognitionUsage(audio_duration=duration),
        )
        self._event_ch.send_nowait(usage_event)

    async def _run(self) -> None:
        """Run the streaming transcription session"""
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            await self._send_task_event.wait()

            # Buffer audio into chunks (50ms chunks)
            samples_50ms = self._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_50ms,
            )

            has_ended = False
            async for data in self._input_ch:
                # Write audio bytes to buffer and get 50ms frames
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())
                    has_ended = True

                for frame in frames:
                    self._audio_duration_collector.push(frame.duration)
                    await ws.send_bytes(frame.data.tobytes())

                    if has_ended:
                        self._audio_duration_collector.flush()
                        has_ended = False

            closing_ws = True

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            while True:
                msg = await ws.receive()

                if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._session.closed:
                        return
                    raise APIStatusError(
                        message="Blaze STT connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Blaze STT message type %s", msg.type)
                    continue

                try:
                    parsed = json.loads(msg.data)
                    self._process_stream_event(parsed)
                except Exception:
                    logger.exception("failed to process Blaze STT message")

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                ws = await self._connect_ws()
                await self._send_config(ws)
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._send_task_event.clear()
                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    tasks_group.exception()  # Retrieve exception to prevent it from being logged
            finally:
                if self._speaking:
                    self._event_ch.send_nowait(stt.SpeechEvent(type=SpeechEventType.END_OF_SPEECH))
                    self._speaking = False

                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        # Convert HTTPS URL to WSS
        base_url = self._opts.base_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{base_url}/realtime"

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    ws_url,
                ),
                self._conn_options.timeout,
            )
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("Failed to connect to Blaze") from e

        return ws

    def _process_stream_event(self, data: dict) -> None:
        """Process incoming WebSocket messages from Blaze."""
        message_type = data.get("type")
        text = data.get("text", "")

        normalized_language = LanguageCode(self._language) if self._language else LanguageCode("en")

        speech_data = stt.SpeechData(
            language=normalized_language,
            text=text,
        )

        if message_type == "ready":
            logger.debug("Received message type ready: %s", data)
            self._send_task_event.set()

        elif message_type == "partial":
            logger.debug("Received message type partial: %s", data)

            if text:
                # Send START_OF_SPEECH if we're not already speaking
                if not self._speaking:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                    )
                    self._speaking = True

                # Send INTERIM_TRANSCRIPT
                interim_event = stt.SpeechEvent(
                    type=SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=[speech_data],
                )
                self._event_ch.send_nowait(interim_event)

        elif message_type == "final":
            # Final committed transcripts - these are sent to the LLM/TTS layer in LiveKit agents
            # and trigger agent responses (unlike partial transcripts which are UI-only)
            if text:
                # Send START_OF_SPEECH if we're not already speaking
                if not self._speaking:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                    )
                    self._speaking = True

                # Send FINAL_TRANSCRIPT but keep speaking=True
                # Multiple commits can occur within the same speech segment
                final_event = stt.SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[speech_data],
                )
                self._event_ch.send_nowait(final_event)


        # Error handling for known Blaze error types
        elif message_type == "error":
            error_msg = data.get("text", "Unknown error")
            logger.error(
                "Blaze STT error [%s]: %s",
                message_type,
                error_msg,
            )
            raise APIConnectionError(f"{message_type}: {error_msg}")
        elif message_type == "successful-authentication":
            logger.debug("Received message type successful-authentication: %s", data)
        else:
            logger.warning("Blaze STT unknown message type: %s, data: %s", message_type, data)

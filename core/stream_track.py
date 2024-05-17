import asyncio
import asyncio
import errno
import fractions
import logging
import threading
import time
from typing import Dict, Optional, Set, Union, Tuple
import av
import numpy as np
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, VideoFrame
from av.audio import AudioStream
from av.frame import Frame
from av.packet import Packet
from av.video.stream import VideoStream
from aiortc import MediaStreamTrack

AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 25  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)


class PlayerStreamTrack(MediaStreamTrack):
    def __init__(self, player, kind, streams=None, block_mode=True):
        super().__init__()
        self.kind = kind
        self.block_mode = block_mode
        self._player = player
        self._queue = asyncio.Queue()
        self._streams = streams
        self.stream_len = len(streams) if streams else 0
        self.train_data_idx = 0
        # self.read_from_render = False
        self.laps = 0
        self.insert_frame_laps = -1
        self.insert_frame_idx = -1
        self.end_frame_laps = -1
        self.end_frame_idx = -1
        self.no_play_frame_nums = 0
        self.has_block = False
        self.blocks = []
        self.audio_chunk = 320  # 16000/(25fps*2)
        self.n_frame = 0

        # 默认空音频
        audio_frame = np.zeros(self.audio_chunk, dtype=np.float32)
        audio_frame = (audio_frame * 32767).astype(np.int16)  # 16bit
        self.default_audio_frame = AudioFrame(format='s16', layout='mono', samples=320)  # 16000/fps
        self.default_audio_frame.planes[0].update(audio_frame.tobytes())
        self.default_audio_frame.sample_rate = 16000

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise MediaStreamError

        if self.kind == 'audio':
            if hasattr(self, "_timestamp"):
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                wait = self._start + (self._timestamp / SAMPLE_RATE) - time.time()
                await asyncio.sleep(wait)

            else:
                self._start = time.time()
                self._timestamp = 0
            return self._timestamp, AUDIO_TIME_BASE

        else:
            # video
            if hasattr(self, "_timestamp"):
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
                await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
            return self._timestamp, VIDEO_TIME_BASE

    async def recv(self) -> Union[Frame, Packet]:
        if self.readyState != "live":
            raise MediaStreamError

        self._player._start(self)

        if self.block_mode:
            if self.kind == 'video':
                if len(self._streams) < self.train_data_idx + 1:
                    self.train_data_idx = 0
                    if self.has_block:
                        self.laps += 1

                if not self.has_block and self.blocks:
                    # 取一个block 初始化状态
                    self.insert_frame_laps, self.insert_frame_idx, self.end_frame_laps, self.end_frame_idx, self.n_frame = self.blocks.pop(0)
                    self.laps = 0
                    self.has_block = True

                if self.has_block and self.laps == self.insert_frame_laps and self.train_data_idx == self.insert_frame_idx:
                    self._player.read_from_render = True
                    self._player.block_audio_frames = self.n_frame * 2

                if self.has_block and (self.laps > self.end_frame_laps or (self.laps == self.end_frame_laps and self.train_data_idx >= self.end_frame_idx)):
                    self._player.read_from_render = False
                    self.has_block = False

                if self._player.read_from_render:
                    frame = await self._queue.get()
                    self.no_play_frame_nums -= 1
                else:
                    frame = self._streams[self.train_data_idx]  # 默认播放talker的模板视频

            else:
                # audio
                if self._player.block_audio_frames:    # use video read from render not accurate
                    try:
                        frame = await asyncio.wait_for(self._queue.get(), 1)  # 阻塞会影响timestamp计算导致卡顿
                    except asyncio.TimeoutError:
                        frame = self.default_audio_frame
                    self._player.block_audio_frames -= 1
                else:
                    frame = self.default_audio_frame
        else:
            frame = await self._queue.get()

        if frame is None:
            self.stop()
            print('stop')
            raise MediaStreamError

        pts, time_base = await self.next_timestamp()
        # frame = VideoFrame(width=640, height=480)
        # for p in frame.planes:
        #     p.update(bytes(p.buffer_size))
        frame.pts = pts
        frame.time_base = time_base
        # if self.kind == 'audio':
        #     frame.sample_rate = SAMPLE_RATE
        if self.block_mode:
            self.train_data_idx += 1
        return frame

    def stop(self):
        print('stop !')
        super().stop()
        if self._player is not None:
            self._player._stop(self)
            self._player = None


def player_worker(
        loop,
        talker_link,
        # audio_track,
        # video_track,
        quit_event,
):
    if talker_link.opt.block_mode:
        # uncoupled
        talker_link.listen_and_calculate_block(quit_event, loop)
    else:
        talker_link.render(quit_event, loop)


class MetaHumanPlayer:  # MediaPlayer
    """A media source that reads audio and/or video from a file."""

    def __init__(
            self, talker
    ):
        self.__talker_link = talker  # talker product video/audio frame
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None

        # examine streams
        self.__started: Set[PlayerStreamTrack] = set()
        # self.__streams = []
        self.__audio: Optional[PlayerStreamTrack] = None
        self.__video: Optional[PlayerStreamTrack] = None

        self.read_from_render = False
        self.block_audio_frames = 0
        streams = talker.get_video_stream() if talker.opt.block_mode else None
        self.__audio = PlayerStreamTrack(self, kind="audio", block_mode=talker.opt.block_mode)
        self.__video = PlayerStreamTrack(self, kind="video", streams=streams, block_mode=talker.opt.block_mode)

    @property
    def audio(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains audio.
        """
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains video.
        """
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        #  when recv offer
        self.__started.add(track)
        self.__talker_link.audio_track = self.__audio
        self.__talker_link.video_track = self.__video

        if self.__thread is None:
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="metahuman-player",
                target=player_worker,
                args=(
                    asyncio.get_event_loop(),
                    self.__talker_link,
                    # self.__audio,
                    # self.__video,
                    self.__thread_quit,
                ),
            )
            self.__thread.start()

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)

        if not self.__started and self.__thread is not None:
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None

        if not self.__started and self.__talker_link is not None:
            self.__talker_link = None

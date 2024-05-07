import asyncio
import audioop
import math
import tempfile
import time
import wave
from threading import Thread

import edge_tts
import numpy as np
import pyaudio

from ai_module.ASR.funasr_client import FunASR
from ai_module.nlp.nlp_ollama_api import question

# 启动时间 (秒)
START_TIME = 0.2
# 释放时间 (秒)
release_time = 0.75


class MikeListener(object):
    def __init__(self, device="", link=None, tts_type=None):
        self.device = device
        self.history_level = []
        self.history_data = []
        self.dynamic_threshold = 0.5  # 声音识别的音量阈值
        self.MAX_LEVEL = 25000
        self.MAX_BLOCK = 100

        self.channels = 1
        self.running = False
        self.processing = False

        self.asr_client = self.new_asrclient()


        # nerf
        self.link = link
        self.tts_type = tts_type

    def start(self):
        Thread(target=self.accept).start()

    def stop(self):
        self.running = False
        self.asr_client.end()

    def new_asrclient(self):
        asrcli = FunASR()
        return asrcli


    def get_history_average(self, number):
        total = 0
        num = 0
        for i in range(len(self.history_level) - 1, -1, -1):
            level = self.history_level[i]
            total += level
            num += 1
            if num >= number:
                break
        return total / num

    async def push(self, voicename: str, text: str, render):
        communicate = edge_tts.Communicate(text, voicename)

        # with open(OUTPUT_FILE, "wb") as file:
        first = True
        async for chunk in communicate.stream():
            if first:
                # render.before_push_audio()
                first = False
            if chunk["type"] == "audio":
                render.push_audio(chunk["data"])
                # file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                pass

    def get_history_percentage(self, number):
        return (self.get_history_average(number) / self.MAX_LEVEL) * 1.05 + 0.02

    def check_device_available(self, p):
        for i in range(p.get_device_count()):
            devInfo = p.get_device_info_by_index(i)
            if devInfo['name'].find(self.device) >= 0 and devInfo['hostApi'] == 0:
                self.channels = devInfo['maxInputChannels']
                return i, devInfo
        return -1, None

    def pyaudio_clear(self):
        while self.running:
            time.sleep(30)

    def save_buffer_to_file(self, buffer):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="cache_data")
        wf = wave.open(temp_file.name, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(buffer)
        wf.close()
        return temp_file.name

    def get_stream(self):
        self.paudio = pyaudio.PyAudio()
        device_id, devInfo = self.check_device_available(self.paudio)
        if device_id < 0:
            print('[!] 无法找到内录设备!')
            return
        channels = int(devInfo['maxInputChannels'])
        if channels == 0:
            print('请检查设备是否有误，再重新启动!')
            return
        stream_res = self.paudio.open(input_device_index=device_id, rate=16000, format=pyaudio.paInt16,
                                      channels=channels, input=True)
        self.running = True
        Thread(target=self.pyaudio_clear).start()  # 等待监听
        return stream_res

    def wait_result(self, asr_client, audio_data):
        self.processing = True
        t = time.time()
        tm = time.time()
        file_url = self.save_buffer_to_file(audio_data)
        self.asr_client.send_url(file_url)

        while not asr_client.done and time.time() - t < 1:  # 超时未识别出语音
            time.sleep(0.01)

        text = asr_client.finalResults
        print( "语音处理完成！ 耗时: {} ms".format(math.floor((time.time() - tm) * 1000)))
        print(text)
        return text



    def accept(self):
        print('开启录音服务...')
        try:
            stream = self.get_stream()
        except Exception as e:
            print(e)
            print("请检查设备是否有误，再重新启动!")

        # 获得麦克风输入流
        is_speaking = False
        last_mute_time = time.time()
        last_speaking_time = time.time()
        concatenated_audio = bytearray()
        while self.running:
            try:
                data = stream.read(1024, exception_on_overflow=False)
            except Exception as e:
                data = None
                print("请检查设备是否有误，再重新启动!")
                return
            if not data:
                continue

            # 电脑麦克风时
            # 只获取第一声道
            data = np.frombuffer(data, dtype=np.int16)
            data = np.reshape(data, (-1, self.channels))  # reshaping the array to split the channels
            mono = data[:, 0]  # taking the first channel
            data = mono.tobytes()

            level = audioop.rms(data, 2)
            if len(self.history_data) >= 5:
                self.history_data.pop(0)
            if len(self.history_level) >= 500:
                self.history_level.pop(0)
            self.history_data.append(data)
            self.history_level.append(level)
            percentage = level / self.MAX_LEVEL
            history_percentage = self.get_history_percentage(30)

            # 根据历史声音动态调节监听音量
            if history_percentage > self.dynamic_threshold:
                self.dynamic_threshold += (history_percentage - self.dynamic_threshold) * 0.0025
            elif history_percentage < self.dynamic_threshold:
                self.dynamic_threshold += (history_percentage - self.dynamic_threshold) * 1

            soon = False
            can_listen = True  # 是否可以监听
            if percentage > self.dynamic_threshold and can_listen:  # 当前音量大于监听的音量
                last_speaking_time = time.time()
                if not self.processing and not is_speaking and time.time() - last_mute_time > START_TIME and len(self.link.asr.queue.queue) <= 0:
                    soon = True  #
                    is_speaking = True  # 用户正在说话
                    print("聆听中...")
                    concatenated_audio.clear()  # 写入前先清空
                    self.asr_client = self.new_asrclient()
                    try:
                        self.asr_client.start()
                    except Exception as e:
                        print(e)
                    for buf in self.history_data:
                        concatenated_audio.extend(buf)

            else:
                # 音量小于监听音量的时候
                last_mute_time = time.time()
                if is_speaking:  # 若当前用户是在说话的, 且间隔时间大于释放时间, 此时意味着此时说话结束
                    if time.time() - last_speaking_time > release_time:
                        is_speaking = False
                        self.asr_client.end()
                        print("语音处理中...")
                        asr_text = self.wait_result(self.asr_client, concatenated_audio)
                        if len(asr_text) < 0:
                            self.dynamic_threshold = self.get_history_percentage(30)
                            self.processing = False
                            print("[!] 语音未检测到内容！")
                        else:
                            response_text = question(asr_text)
                            if self.link and self.tts_type:
                                self.link.say(response_text, self.tts_type)

            if not soon and is_speaking:
                concatenated_audio.extend(data)


if __name__ == '__main__':
    print('开启录音服务...')
    mike_listener = MikeListener("")  # 监听麦克风
    mike_listener.start()

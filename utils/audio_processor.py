import yt_dlp 
from pydub import AudioSegment
import os

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url:str)-> str:
    output_path = os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s')
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'cookiefile': 'cookies.txt',
        # 'cookiesfrombrowser': ('edge',), 
        'extractor_args':{
            'youtube':{
                'player_client':['default','-android_sdkless']
            }
        },
        'http_headers':{
            'User-Agent':'Mozilla/5.0(Windows NT 10.0; Win64; x64) AppleWebkit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept':'text/html, application/xhtml+xml, application/xml; q=0.9, */*; q-0.8',
            'Accept-Language': 'en-us, en;q=0.5'
        },
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        # "quiet": True,
        "verbose": True,
    }
    print("Cookies file exists:", os.path.exists("cookies.txt"))
    print("Current working directory:", os.getcwd())
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url,download=True)
        filename = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
    return filename

def convert_to_wav(input_path:str)-> str:
    """Convert any audio/video file to WAV format using pydub"""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path

def chunk_audio(wav_path:str, chunk_minutes:int = 10)->list:
    """Chunk a WAV file into smaller segments of specified minutes"""
    audio = AudioSegment.from_wav(wav_path)
    chunk_length_ms = chunk_minutes * 60 * 1000
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk = audio[start:start + chunk_length_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def process_input(source:str)-> list:
    """Process input which can be a YouTube URL or a local file path"""
    if source.startswith("http://") or source.startswith("https://"):
        print(f"Downloading audio from YouTube URL: {source}")
        wav_path = download_youtube_audio(source)
    else:
        print(f"Processing local file: {source}")
        wav_path = convert_to_wav(source)
    print(f"Chunking audio file: {wav_path}")
    chunks = chunk_audio(wav_path)
    return chunks
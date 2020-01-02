from flask import jsonify, request, Flask
from utils import config
from utils.logger import logger
from utils.server_utils import infer
import io
import soundfile as sf
import resampy
import librosa

app = Flask(__name__)

def get_audio_from_request(request):
    audio_file = request.files['audio']
    data = audio_file.read()
    audio, sampling_rate = sf.read(io.BytesIO(data), always_2d=True, dtype='float32')
    if sampling_rate != 16000:
        audio = resampy.resample(audio, sampling_rate, 16000, axis=-1)
    return librosa.effects.trim(audio.reshape(-1))[0]

@app.route('/audio', methods=['POST'])
def audio_post():
    audio = get_audio_from_request(request)
    result = infer(audio)
    logger.info(f"Received utterance from {request.remote_addr}. Transcript was: {result}")
    return jsonify({'text': result})

def main():
    app.run(config.host_ip, config.port)

if __name__ == "__main__":
    main()
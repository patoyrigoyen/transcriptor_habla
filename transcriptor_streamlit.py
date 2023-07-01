import streamlit as st
import numpy as np
import whisper
import contextlib
import wave
from pyannote.audio import Audio
from pyannote.core import Segment
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)
from sklearn.cluster import AgglomerativeClustering
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
import tempfile



def segment_embedding(segment):
  start = segment['start']
  end = min(duration, segment['end'])
  clip = Segment(start, end)
  waveform, sample_rate = audio.crop(path, clip)

  # Convert to mono
  #   waveform = librosa.to_mono(waveform.numpy())
  #   waveform = torch.from_numpy(waveform)
  # Convert to mono
  if waveform.dim() == 2:
      waveform = waveform.mean(dim=0)
  print("WORKED")

  # Add batch and channel dimensions
  waveform = waveform.unsqueeze(0).unsqueeze(0)

  return embedding_model(waveform)




###########################################################################

st.subheader("**Transcribe tus entrevistas (beta)**", )

###########################################################################
with st.container():
    # upload audio file with streamlit
    settings_expander = st.expander(label='Configura el audio')
    col1, padding, col2 = settings_expander.columns((10, 2, 5))
    audio_file = col1.file_uploader("Sube tu archivo de audio", type=["wav"])
    num_speakers = col2.slider("Número de hablantes", min_value=1, max_value=5, value=2, step=1)

    original_audio_expander = st.expander(label='Escucha el audio original')
    original_audio_expander.audio(audio_file)

col1, col2, col3 = st.columns([1,1,1])
button_transcribe = col2.button("Transcribir Audio", type='primary', use_container_width=True)

###########################################################################

if button_transcribe:
    if audio_file is not None:
        st.success("Transcribiendo tu audio, esto puede tomar ~20min...")
        ##path = '/Users/patricio.yrigoyen/Desktop/Transcriptor/'+audio_file.name
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(audio_file.getvalue())
        path = tfile.name

        print("Beginning transcription")
        model = whisper.load_model("large-v2")
        result = model.transcribe(path)
        print("Finished transcription")
        segments = result['segments']
        print("Stored transcription succesfully...")

        with contextlib.closing(wave.open(path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames/float(rate)

        audio = Audio()

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)

        embeddings = np.nan_to_num(embeddings)

        #identify speakers
        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_

        with contextlib.closing(wave.open(path, 'r')) as f:
            # Read the entire file into a numpy array
            audio = np.frombuffer(f.readframes(-1), np.int16)
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        # st.text_input('Identifiquemos a los hablantes, presiona 0 para continuar...', '')
        speaker = ['0']*num_speakers
        input_val = '0'

        for i in range(num_speakers):
            sp = [x for x in segments if x['speaker'] == f'SPEAKER {i+1}']
            j =-1
            while input_val=='0':
                j += 1
                seg = sp[j]
                start_time = seg['start']
                end_time = seg['end']
                start_sample = int(start_time * f.getframerate())
                end_sample = int(end_time * f.getframerate())
                cropped_audio = audio[start_sample:end_sample]

                # Play the cropped audio
                start_ms = start_time * 1000
                end_ms = end_time * 1000
                sound = AudioSegment.from_file(path, format="wav")
                splice = sound[start_ms:end_ms]

                # Get speaker name
                input_val = st.text_input('Identifica al hablante ' + str(i+1) + ':', '')
                buffer = BytesIO()
                splice.export(buffer, format='wav')
                buffer.seek(0)
                st.audio(buffer, format='audio/wav')

                if input_val!=0:
                    speaker[i] = input_val
            input_val = '0'

        # os.remove(path)

        st.markdown(f'Los hablantes identificados son: speaker')
        for i in range(len(segments)):
            segments[i]["speaker"] = speaker[labels[i]]

        #write transcript
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                st.markdown("**\n" + segment["speaker"] + ': **')
            st.markdown(segment["text"][1:] + ' ')

    else:
        st.error("Por favor carga un archivo primero")

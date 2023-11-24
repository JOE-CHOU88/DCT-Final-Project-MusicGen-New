import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write, audio_read

print("-------Start Model Loaded-------")
model = MusicGen.get_pretrained('facebook/musicgen-melody')
print("-------Done Model Loaded-------")

model.set_generation_params(duration=15)  # generate 15 seconds.

base_audio, sr = audio_read("./assets/bach.mp3")
duration = 3
base_audio = base_audio[:, :sr*duration]
# print(base_audio.shape)

print("-------Start Stylizing Music-------")
melody_version = model.generate_with_chroma(descriptions=["happy atmosphere"], melody_wavs=base_audio, melody_sample_rate=sr)
print("-------Done Stylizing Music-------")
audio_write("output-audio/bach_output", melody_version[0].cpu(), sample_rate=model.sample_rate)
print("-------Finish-------")
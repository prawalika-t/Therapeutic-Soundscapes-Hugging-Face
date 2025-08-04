import gradio as gr
import torch
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from pydub import AudioSegment
import tempfile

# Load model and processor
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Mood to Valence-Arousal mapping
mood_va_map = {
    "anxious": (0.2, 0.8),
    "calm": (0.8, 0.2),
    "happy": (0.9, 0.9),
    "sad": (0.1, 0.1),
    "angry": (0.1, 0.9),
    "relaxed": (0.7, 0.3),
    "fearful": (0.2, 0.9),
    "peaceful": (0.9, 0.2),
}

def infer_mood_from_hr(hr):
    if hr > 95:
        return "anxious"
    elif hr > 85:
        return "fearful"
    elif hr > 75:
        return "angry"
    elif hr > 70:
        return "happy"
    elif hr > 65:
        return "sad"
    elif hr > 55:
        return "relaxed"
    else:
        return "calm"

def find_opposite_mood(mood):
    val, ar = mood_va_map.get(mood, (0.5, 0.5))
    target = (1 - val, 1 - ar)
    min_dist = float('inf')
    opposite = mood
    for m, (v, a) in mood_va_map.items():
        dist = np.linalg.norm(np.array([v, a]) - np.array(target))
        if dist < min_dist:
            min_dist = dist
            opposite = m
    return opposite

def suggest_genre(mood):
    return {
        "calm": "ambient",
        "relaxed": "lofi",
        "happy": "pop",
        "sad": "classical",
        "anxious": "jazz",
        "angry": "metal",
        "fearful": "cinematic",
        "peaceful": "ambient",
    }.get(mood, "ambient")

def generate_prompt(start_mood, target_mood, genre):
    return (
        f"Transition from {start_mood} to {target_mood} mood using {genre} instrumental music. "
        f"Gradual emotional evolution."
    )

def numpy_to_audiosegment(np_audio, sample_rate):
    audio_16bit = (np_audio * 32767).astype(np.int16)
    return AudioSegment(
        audio_16bit.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )

def generate_music(hr, genre):
    start_mood = infer_mood_from_hr(hr)
    target_mood = find_opposite_mood(start_mood)
    prompt = generate_prompt(start_mood, target_mood, genre)

    inputs = processor(text=[prompt], return_tensors="pt").to(model.device)
    audio_values = model.generate(**inputs, max_new_tokens=2500)
    audio_array = audio_values[0].cpu().numpy().flatten()
    sampling_rate = model.config.audio_encoder.sampling_rate
    final_music = numpy_to_audiosegment(audio_array, sampling_rate)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        final_music.export(tmpfile.name, format="wav")
        return tmpfile.name, start_mood, target_mood, prompt

iface = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Slider(40, 120, value=72, label="Heart Rate (bpm)"),
        gr.Dropdown(["ambient", "pop", "jazz", "classical", "metal", "lofi", "cinematic"], label="Genre")
    ],
    outputs=[
        gr.Audio(label="Generated Music"),
        gr.Text(label="Detected Mood"),
        gr.Text(label="Target Mood"),
        gr.Text(label="Prompt")
    ],
    title="🧠 Mood-Aware Music Generator (ISO Principle)",
    description="Enter your heart rate and preferred genre to receive music designed to shift your mood using the ISO Principle."
)

if __name__ == "__main__":
    iface.launch()
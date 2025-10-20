from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import soundfile as sf
from gtts import gTTS
from pydub import AudioSegment
import streamlit as st 
from PIL import Image
load_dotenv(find_dotenv())



def img2text(url):
    image = Image.open(url)
    image_to_text = pipeline("image-to-text", model = "Salesforce/blip-image-captioning-base")
    text = image_to_text(image)[0]["generated_text"]
    print(text)
    return text


def text2story(prompt):
    story_generator = pipeline(
        "text-generation",
        model="gpt2",
        max_length=300,
        num_return_sequences=1
    )
    output = story_generator(
        f"Write a short story based on this prompt: {prompt}"
    )[0]["generated_text"]
    story = output.replace(f"Write a short story based on this prompt: {prompt}", "").strip()
    print("\nGenerated Story:\n", story)
    return story




def text2speech(text, output_filename):
    try:
        tts = gTTS(text=text, lang='en')
        temp_mp3 = "temp_audio.mp3"
        tts.save(temp_mp3)
        audio = AudioSegment.from_mp3(temp_mp3)
        audio.export(output_filename, format="wav")
        print(f"\nText successfully converted to speech and saved to {output_filename}")
    except Exception as e:
        print(f"An error occurred during text-to-speech generation: {e}")



st.title("Image → Story → Audio Generator")
uploaded_image = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Story from Image"):
        with st.spinner("Extracting text from image..."):
            caption = img2text(uploaded_image)
            st.write("**📝 Image Description:**", caption)

        with st.spinner("Generating story..."):
            story = text2story(caption)
            st.write("**📖 Generated Story:**")
            st.write(story)

        with st.spinner("Converting story to audio..."):
            audio_file = text2speech(story,"output.wav")

        st.audio(audio_file, format="audio/wav")

        if audio_file is not None:
                with open(audio_file, "rb") as f:
                   
                   pass


        else:
           pass






import pyttsx3
# from gtts import gTTS # this is used for saving audio file

# Initialize the TTS engine
engine = pyttsx3.init()

# Get text input from the user
text = "my name is D krishna, message: i am in danger."

# Set properties before adding anything to speak
engine.setProperty('rate', 150)    # Speed of speech
engine.setProperty('volume', 1)     # Volume (0.0 to 1.0)

# Convert text to speech
engine.say(text)

# Wait until the speech is finished
engine.runAndWait()

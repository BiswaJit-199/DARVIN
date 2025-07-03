# voice_alert.py
import pyttsx3
from threading import Thread
from time import time, localtime, strftime
from utils.constants import ALERT_COOLDOWN

engine = pyttsx3.init()
engine.setProperty('rate', 150)

_last_spoken_time = 0

def speak(text):
	global _last_spoken_time
	now = time()

	if now - _last_spoken_time < ALERT_COOLDOWN:
		return #Skip alert if it's too soon

	def _speak():
		try:
			engine.say(text)
			engine.runAndWait()
			print(f"Voice Alert Triggered: {text} - at {strftime('%Y-%m-%d %H:%M:%S', localtime(now))}")
		except Exception as e:
			print(f"Error speaking text: {e}")

	_last_spoken_time = now
	Thread(target=_speak).start()
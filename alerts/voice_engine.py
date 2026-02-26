"""
Voice alert system for fatigue detection
"""
import pyttsx3
import threading
from datetime import datetime


class VoiceAlertEngine:
    """Text-to-speech alert system"""

    def __init__(self, rate=150, volume=1.0):
        """
        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

        # Alert cooldown to prevent spam
        self.last_alert_time = None
        self.cooldown_seconds = 10

    def speak(self, text, async_mode=True):
        """
        Speak the given text

        Args:
            text: Text to speak
            async_mode: If True, speak in a separate thread
        """
        if async_mode:
            thread = threading.Thread(target=self._speak_sync, args=(text,))
            thread.daemon = True
            thread.start()
        else:
            self._speak_sync(text)

    def _speak_sync(self, text):
        """Synchronous speech"""
        self.engine.say(text)
        self.engine.runAndWait()

    def alert_drowsy(self):
        """Alert for drowsy state"""
        if self._can_alert():
            self.speak("Warning! You appear drowsy. Please take a break.")
            self.last_alert_time = datetime.now()

    def alert_very_drowsy(self):
        """Alert for very drowsy state"""
        if self._can_alert():
            self.speak("Danger! Severe fatigue detected. Please stop and rest immediately.")
            self.last_alert_time = datetime.now()

    def _can_alert(self):
        """Check if enough time has passed since last alert"""
        if self.last_alert_time is None:
            return True

        elapsed = (datetime.now() - self.last_alert_time).total_seconds()
        return elapsed >= self.cooldown_seconds

    def set_voice(self, voice_id=None):
        """
        Set voice for TTS

        Args:
            voice_id: Voice ID (None for default)
        """
        voices = self.engine.getProperty('voices')
        if voice_id is not None and voice_id < len(voices):
            self.engine.setProperty('voice', voices[voice_id].id)

    def list_voices(self):
        """List available voices"""
        voices = self.engine.getProperty('voices')
        for i, voice in enumerate(voices):
            print(f"{i}: {voice.name} - {voice.languages}")


if __name__ == "__main__":
    # Test voice engine
    engine = VoiceAlertEngine()
    engine.list_voices()
    engine.speak("Fatigue detection system initialized", async_mode=False)

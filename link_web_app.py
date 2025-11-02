from face_emotions import EmotionRecognizer


# Example usage: call this function from a separate admin script or route to reload model after training


def reload_recognizer(app_instance, model_path='models/emotion_model.h5'):
app_instance.recognizer = EmotionRecognizer(model_path=model_path)
return app_instance.recognizer
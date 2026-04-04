class Credibility:
    def evaluate(self, features):
        handlers = {
            "image": self._image,
            "video": self._video,
            "document": self._document,
        }

        handler = handlers.get(features["type"])
        return handler(features) if handler else self._default()

    def _image(self, f):
        return self._result(0.8, "real", "Image looks authentic")

    def _video(self, f):
        return self._result(0.7, "uncertain", "Video unclear")

    def _document(self, f):
        return self._result(0.9, "real", "Document valid")

    def _default(self):
        return self._result(0.5, "unknown", "No handler")

    def _result(self, score, label, explanation):
        return {
            "score": score,
            "label": label,
            "explanation": explanation,
            "flags": []
        }
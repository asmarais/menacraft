class Preprocessing:
    def process(self, input_type, data):
        if input_type == "image":
            return self._image(data)
        elif input_type == "video":
            return self._video(data)
        elif input_type == "document":
            return self._document(data)
        elif input_type == "url":
            return self._url(data)

    def _image(self, data):
        return {"type": "image", "features": {}}

    def _video(self, data):
        return {"type": "video", "features": {}}

    def _document(self, data):
        return {"type": "document", "features": {}}

    def _url(self, data):
        return {"type": "url", "features": {}}
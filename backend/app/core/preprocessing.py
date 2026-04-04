from .preprocess_image import preprocess_image
from .preprocess_video import preprocess_video
from .preprocess_document import preprocess_document
from .preprocess_url import preprocess_url

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
        return preprocess_image(data)

    def _video(self, data):
        return preprocess_video(data)

    def _document(self, data):
        return preprocess_document(data)

    def _url(self, data):
        return preprocess_url(data)
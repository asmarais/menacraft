from app.core.preprocessing import Preprocessing
from app.axes.authenticity import Authenticity
from app.axes.consistency import Consistency
from app.axes.credibility import Credibility

class Pipeline:
    def __init__(self):
        self.pre = Preprocessing()
        self.auth = Authenticity()
        self.cons = Consistency()
        self.cred = Credibility()

    def run(self, input_type, data):
        features = self.pre.process(input_type, data)

        return {
            "authenticity": self.auth.evaluate(features),
            "consistency": self.cons.evaluate(features),
            "credibility": self.cred.evaluate(features),
        }
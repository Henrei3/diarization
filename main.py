
from pyannote.audio import Pipeline
from pydub import AudioSegment
from datetime import datetime
from time import sleep
before = datetime.now()

token = "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"

path = "./out-0980427196-702-20230427-092354-1682605434.259541.wav"

pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1', use_auth_token=token)

dz = pipeline(path)

print(dz)

with open("./diarization.txt", "w") as diarization:
    dz.write_rttm(diarization)

with open("./diarization2.txt", "w") as text_file:
    text_file.write(str(dz))

print(*list(dz.itertracks(yield_label=True))[:10], sep="\n")


print("Execution time =  " + str(datetime.now()-before))

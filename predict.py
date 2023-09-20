# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
from cog import BasePredictor, Input, Path
import torch
import whisperx
import json
import uuid
import yt_dlp


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        os.environ["HF_HOME"] = "/src/hf_models"
        os.environ["TORCH_HOME"] = "/src/torch_models"
        self.device = "cuda"
        self.compute_type = "float16"
        self.model = whisperx.load_model(
            "large-v2", self.device, language="en", compute_type=self.compute_type
        )

    def predict(
        self,
        url: str = Input(
            description="Video URL. View supported sites https://dub.sh/supportedsites"
        ),
        batch_size: int = Input(
            description="Parallelization of input audio transcription", default=16
        ),
        debug: bool = Input(
            description="Print out memory usage information.", default=False
        ),
    ) -> str:
        """Run a single prediction on the model"""
        try:
            with torch.inference_mode():
                rand_id = uuid.uuid4().hex
                ydl_opts = {
                    "format": "bestaudio/best",
                    "postprocessors": [
                        {
                            "key": "FFmpegExtractAudio",
                            "preferredcodec": "mp3",
                        }
                    ],
                    "outtmpl": f"{rand_id}.%(ext)s",
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download(url)

                result = self.model.transcribe(f"{rand_id}.mp3", batch_size=batch_size)

                os.remove(f"{rand_id}.mp3")

                if debug:
                    print(
                        f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB"
                    )
            return json.dumps(
                {
                    "segments": result["segments"],
                }
            )
        except Exception as e:
            raise (e)

"""Download the V-JEPA 2 models once so the talk works without internet."""
from transformers import AutoVideoProcessor, AutoModel, AutoModelForVideoClassification

for mid, cls in [("facebook/vjepa2-vitl-fpc64-256", AutoModel),
                 ("facebook/vjepa2-vitl-fpc16-256-ssv2", AutoModelForVideoClassification)]:
    print("downloading", mid)
    AutoVideoProcessor.from_pretrained(mid)
    cls.from_pretrained(mid)
print("all models cached in ~/.cache/huggingface")

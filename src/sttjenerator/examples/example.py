from sttjenerator.models import registry

config = {
    "model": "faster_whisper",
    "model_path": "medium",
    "sample_rate": 16000,
    "sd_default_device": 7,
}

print(registry.MODEL_REGISTRY)

stt_generator = registry.get_model_class(config)
stt_generator.generate()


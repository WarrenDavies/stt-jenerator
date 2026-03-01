from sttjenerator.models import registry

config = {
    "model_name": "faster-whisper",
    "model_path": "medium",
    "sample_rate": 16000,
    "sd_default_device": 7,
}
base_class = registry.get_model_class(config)
base_class.generate()


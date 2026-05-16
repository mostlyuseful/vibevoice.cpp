#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "convert_vibevoice_to_gguf.py"
SPEC = importlib.util.spec_from_file_location("convert_vibevoice_to_gguf", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)

KUGEL_7B_CFG = json.loads(
    (ROOT / ".." / "kugelaudio-open" / "src" / "kugelaudio_open" / "configs" / "kugelaudio_7b.json").read_text()
)
KUGEL_15B_CFG = json.loads(
    (ROOT / ".." / "kugelaudio-open" / "src" / "kugelaudio_open" / "configs" / "kugelaudio_1.5b.json").read_text()
)


class FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)
        self.dtype = self._arr.dtype

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class FakeSafeOpen:
    def __init__(self, tensors):
        self._tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def keys(self):
        return list(self._tensors.keys())

    def get_tensor(self, key):
        return FakeTensor(self._tensors[key])


class FakeWriter:
    last_instance = None

    def __init__(self, path, arch):
        self.path = path
        self.arch = arch
        self.metadata = {}
        self.tensors = {}
        FakeWriter.last_instance = self

    def add_string(self, key, value):
        self.metadata[key] = value

    def add_uint32(self, key, value):
        self.metadata[key] = value

    def add_float32(self, key, value):
        self.metadata[key] = value

    def add_array(self, key, value):
        self.metadata[key] = list(value)

    def add_tensor(self, key, value):
        self.tensors[key] = np.asarray(value)

    def write_header_to_file(self):
        pass

    def write_kv_data_to_file(self):
        pass

    def write_tensors_to_file(self):
        pass

    def close(self):
        pass


class FakeGGUFModule:
    GGUFWriter = FakeWriter


class ConverterTests(unittest.TestCase):
    def test_detects_supported_kugelaudio_open_signature(self):
        self.assertEqual(MOD.detect_variant(KUGEL_7B_CFG, []), "kugelaudio-0-open")

    def test_rejects_other_kugelaudio_variants_for_v1(self):
        with self.assertRaisesRegex(ValueError, "only kugelaudio/kugelaudio-0-open"):
            MOD.detect_variant(KUGEL_15B_CFG, [])

    def test_convert_emits_explicit_kugelaudio_metadata_contract(self):
        fake_tensors = {
            "model.language_model.embed_tokens.weight": np.zeros((2, 2), dtype=np.float32),
            "model.speech_scaling_factor": np.array(1.0, dtype=np.float32),
            "model.speech_bias_factor": np.array(0.0, dtype=np.float32),
        }

        def fake_safe_open(_path, framework="pt"):
            self.assertEqual(framework, "pt")
            return FakeSafeOpen(fake_tensors)

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            out = Path(td) / "out.gguf"
            src.mkdir()
            (src / "config.json").write_text(json.dumps(KUGEL_7B_CFG))
            (src / "model.safetensors").write_bytes(b"stub")

            rc = MOD.convert_checkpoint(
                src,
                out,
                strict=True,
                dtype="fp32",
                gguf_module=FakeGGUFModule(),
                safe_open_fn=fake_safe_open,
            )

        self.assertEqual(rc, 0)
        writer = FakeWriter.last_instance
        self.assertIsNotNone(writer)
        self.assertEqual(writer.arch, "vibevoice")
        self.assertEqual(writer.metadata["kugelaudio.schema_version"], 1)
        self.assertEqual(writer.metadata["kugelaudio.architecture"], "kugelaudio")
        self.assertEqual(writer.metadata["kugelaudio.checkpoint"], "kugelaudio-0-open")
        self.assertEqual(writer.metadata["kugelaudio.decoder.hidden_size"], 3584)
        self.assertEqual(writer.metadata["kugelaudio.diffusion.latent_size"], 64)
        self.assertEqual(writer.metadata["vibevoice.variant"], "kugelaudio-0-open")
        self.assertIn("lm.tok_embd.weight", writer.tensors)


if __name__ == "__main__":
    unittest.main()

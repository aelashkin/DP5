import os
import platform
import unittest
from importlib import metadata


class NeuralNetImportTests(unittest.TestCase):
    def test_neural_net_runtime_imports(self):
        import tensorflow  # noqa: F401
        import keras  # noqa: F401
        import rdkit  # noqa: F401
        from dp5.neural_net.nn_utils import get_nn_shifts

        self.assertTrue(callable(get_nn_shifts))

    @unittest.skipUnless(
        os.environ.get("DP5_ASSERT_TENSORFLOW_METAL_MARKER") == "1",
        "enabled only in clean platform CI jobs",
    )
    def test_tensorflow_metal_marker_matches_platform(self):
        expected = platform.system() == "Darwin" and platform.machine() == "arm64"

        try:
            metadata.version("tensorflow-metal")
            installed = True
        except metadata.PackageNotFoundError:
            installed = False

        self.assertEqual(installed, expected)

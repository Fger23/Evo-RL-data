"""Private configuration and temporary SSH password authentication tests."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pkl_converter import converter


class PasswordTests(unittest.TestCase):
    def test_key_auth_and_invalid_password(self):
        with converter.ssh_authentication(None) as (options, environment):
            self.assertEqual(options, ["-o", "BatchMode=yes"])
            self.assertIsNone(environment)
        for invalid in ("", 123):
            with self.subTest(value=invalid), self.assertRaises(ValueError), converter.ssh_authentication(invalid):
                pass

    @unittest.skipUnless(os.name == "posix", "SSH_ASKPASS password mode runs on Linux/macOS")
    def test_askpass_uses_environment_and_is_cleaned(self):
        secret = "example-test-password"
        previous = os.environ.get("EVO_SSH_UPLOAD_PASSWORD")
        with converter.ssh_authentication(secret) as (options, environment):
            helper = Path(environment["SSH_ASKPASS"])
            self.assertNotIn(secret, helper.read_text())
            self.assertNotIn(secret, " ".join(options))
            self.assertEqual(helper.stat().st_mode & 0o777, 0o700)
            self.assertIn("BatchMode=no", options)
            self.assertIn("StrictHostKeyChecking=yes", options)
            result = subprocess.run([str(helper)], env=environment, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), secret)
        self.assertFalse(helper.exists())
        self.assertNotIn("EVO_SSH_UPLOAD_PASSWORD", environment)
        self.assertEqual(os.environ.get("EVO_SSH_UPLOAD_PASSWORD"), previous)

    def test_private_config_password_env_override_and_no_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "upload.json"
            config.write_text(json.dumps({"upload_target": "test@example.invalid:/dataset",
                                          "upload_port": 2222, "upload_password": "example-config-password"}))
            with (patch.object(converter, "capture_info", return_value=(root / "raw", {}, [])),
                  patch.object(converter, "convert", return_value=root / "output"),
                  patch.object(converter, "upload") as upload):
                arguments = ["convert", "--input", "raw", "--config", str(config)]
                with patch.object(sys, "argv", arguments):
                    converter.main()
                upload.assert_called_once_with(root / "output", "example.invalid", "test", "/dataset", 2222,
                                               password="example-config-password")
                upload.reset_mock()
                with (patch.dict(os.environ, {"EVO_TEST_PASSWORD": "example-env-password"}),
                      patch.object(sys, "argv", arguments + ["--upload-password-env", "EVO_TEST_PASSWORD"])):
                    converter.main()
                upload.assert_called_once_with(root / "output", "example.invalid", "test", "/dataset", 2222,
                                               password="example-env-password")
                upload.reset_mock()
                with patch.object(sys, "argv", arguments + ["--no-upload"]):
                    converter.main()
                upload.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

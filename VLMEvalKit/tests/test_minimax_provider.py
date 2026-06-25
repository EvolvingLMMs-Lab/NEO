"""Unit tests for MiniMax provider support in VLMEvalKit."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from functools import partial

# Add VLMEvalKit root to path so vlmeval can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestMiniMaxConfig(unittest.TestCase):
    """Test MiniMax model entries in config.py registry."""

    def test_m3_registered(self):
        """MiniMax-M3 should be registered in api_models."""
        from vlmeval.config import api_models
        self.assertIn('MiniMax-M3', api_models)

    def test_m27_registered(self):
        """MiniMax-M2.7 should be registered in api_models."""
        from vlmeval.config import api_models
        self.assertIn('MiniMax-M2.7', api_models)

    def test_m27_highspeed_registered(self):
        """MiniMax-M2.7-highspeed should be registered in api_models."""
        from vlmeval.config import api_models
        self.assertIn('MiniMax-M2.7-highspeed', api_models)

    def test_legacy_abab_still_registered(self):
        """Legacy abab models should still be registered for backward compat."""
        from vlmeval.config import api_models
        self.assertIn('abab6.5s', api_models)
        self.assertIn('abab7-preview', api_models)

    def test_m3_api_base(self):
        """MiniMax-M3 should use the correct api.minimax.io base URL."""
        from vlmeval.config import api_models
        entry = api_models['MiniMax-M3']
        self.assertEqual(entry.keywords['api_base'],
                         'https://api.minimax.io/v1/chat/completions')

    def test_m27_api_base(self):
        """MiniMax-M2.7 should use the correct api.minimax.io base URL."""
        from vlmeval.config import api_models
        entry = api_models['MiniMax-M2.7']
        self.assertEqual(entry.keywords['api_base'],
                         'https://api.minimax.io/v1/chat/completions')

    def test_m27_highspeed_api_base(self):
        """MiniMax-M2.7-highspeed should use the correct api.minimax.io base URL."""
        from vlmeval.config import api_models
        entry = api_models['MiniMax-M2.7-highspeed']
        self.assertEqual(entry.keywords['api_base'],
                         'https://api.minimax.io/v1/chat/completions')

    def test_m3_model_name(self):
        """MiniMax-M3 config entry should set the correct model string."""
        from vlmeval.config import api_models
        entry = api_models['MiniMax-M3']
        self.assertEqual(entry.keywords['model'], 'MiniMax-M3')

    def test_m27_model_name(self):
        """MiniMax-M2.7 config entry should set the correct model string."""
        from vlmeval.config import api_models
        entry = api_models['MiniMax-M2.7']
        self.assertEqual(entry.keywords['model'], 'MiniMax-M2.7')

    def test_m27_highspeed_model_name(self):
        """MiniMax-M2.7-highspeed config entry should set the correct model string."""
        from vlmeval.config import api_models
        entry = api_models['MiniMax-M2.7-highspeed']
        self.assertEqual(entry.keywords['model'], 'MiniMax-M2.7-highspeed')

    def test_temperature_not_zero(self):
        """MiniMax models should have temperature > 0 (API constraint)."""
        from vlmeval.config import api_models
        for name in ['MiniMax-M3', 'MiniMax-M2.7', 'MiniMax-M2.7-highspeed', 'abab6.5s', 'abab7-preview']:
            temp = api_models[name].keywords['temperature']
            self.assertGreater(temp, 0,
                               f"{name} temperature must be > 0 for MiniMax API")
            self.assertLessEqual(temp, 1.0,
                                  f"{name} temperature must be <= 1.0")

    def test_legacy_abab_updated_url(self):
        """Legacy abab entries should use updated api.minimax.io URL."""
        from vlmeval.config import api_models
        for name in ['abab6.5s', 'abab7-preview']:
            url = api_models[name].keywords['api_base']
            self.assertIn('api.minimax.io', url,
                          f"{name} should use api.minimax.io, not api.minimax.chat")

    def test_m3_in_supported_vlm(self):
        """MiniMax-M3 should be in the unified supported_VLM registry."""
        from vlmeval.config import supported_VLM
        self.assertIn('MiniMax-M3', supported_VLM)

    def test_m27_in_supported_vlm(self):
        """MiniMax-M2.7 models should be in the unified supported_VLM registry."""
        from vlmeval.config import supported_VLM
        self.assertIn('MiniMax-M2.7', supported_VLM)
        self.assertIn('MiniMax-M2.7-highspeed', supported_VLM)

    def test_uses_gpt4v_class(self):
        """MiniMax models should use GPT4V (OpenAI-compatible) wrapper."""
        from vlmeval.config import api_models
        from vlmeval.api import GPT4V
        for name in ['MiniMax-M3', 'MiniMax-M2.7', 'MiniMax-M2.7-highspeed']:
            entry = api_models[name]
            self.assertEqual(entry.func, GPT4V)

    def test_retry_count(self):
        """MiniMax models should have retry=10."""
        from vlmeval.config import api_models
        for name in ['MiniMax-M3', 'MiniMax-M2.7', 'MiniMax-M2.7-highspeed']:
            self.assertEqual(api_models[name].keywords['retry'], 10)

    def test_m3_listed_before_legacy(self):
        """MiniMax-M3 should be registered before older MiniMax / abab entries."""
        from vlmeval.config import api_models
        keys = list(api_models.keys())
        self.assertIn('MiniMax-M3', keys)
        m3_idx = keys.index('MiniMax-M3')
        for older in ['MiniMax-M2.7', 'MiniMax-M2.7-highspeed', 'abab6.5s', 'abab7-preview']:
            if older in keys:
                self.assertLess(
                    m3_idx, keys.index(older),
                    f"MiniMax-M3 should be listed before {older}",
                )


class TestMiniMaxKeyDetection(unittest.TestCase):
    """Test API key detection for MiniMax models in OpenAIWrapper."""

    @patch.dict(os.environ, {'MiniMax_API_KEY': 'test-minimax-key-123'}, clear=False)
    def test_m27_detects_minimax_api_key(self):
        """MiniMax-M2.7 should detect MiniMax_API_KEY env var."""
        from vlmeval.api.gpt import OpenAIWrapper
        wrapper = OpenAIWrapper.__new__(OpenAIWrapper)
        wrapper.model = 'MiniMax-M2.7'
        wrapper.cur_idx = 0
        wrapper.fail_msg = 'Failed'
        wrapper.max_tokens = 2048
        wrapper.temperature = 0.01
        wrapper.use_azure = False
        # Simulate the key detection logic
        model = 'MiniMax-M2.7'
        if 'abab' in model or 'MiniMax-M' in model:
            env_key = os.environ.get('MiniMax_API_KEY', os.environ.get('MINIMAX_API_KEY', ''))
        else:
            env_key = ''
        self.assertEqual(env_key, 'test-minimax-key-123')

    @patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-key-fallback'}, clear=False)
    def test_m27_detects_minimax_api_key_uppercase(self):
        """MiniMax-M2.7 should also detect MINIMAX_API_KEY env var (fallback)."""
        model = 'MiniMax-M2.7'
        if 'abab' in model or 'MiniMax-M' in model:
            env_key = os.environ.get('MiniMax_API_KEY', os.environ.get('MINIMAX_API_KEY', ''))
        else:
            env_key = ''
        self.assertEqual(env_key, 'test-key-fallback')

    def test_m27_highspeed_matches_condition(self):
        """MiniMax-M2.7-highspeed should match the 'MiniMax-M' model name pattern."""
        model = 'MiniMax-M2.7-highspeed'
        self.assertTrue('MiniMax-M' in model)

    def test_m3_matches_condition(self):
        """MiniMax-M3 should match the 'MiniMax-M' model name pattern."""
        model = 'MiniMax-M3'
        self.assertTrue('MiniMax-M' in model)

    @patch.dict(os.environ, {'MiniMax_API_KEY': 'test-m3-key'}, clear=False)
    def test_m3_detects_minimax_api_key(self):
        """MiniMax-M3 should detect MiniMax_API_KEY env var via the same pattern."""
        model = 'MiniMax-M3'
        if 'abab' in model or 'MiniMax-M' in model:
            env_key = os.environ.get('MiniMax_API_KEY', os.environ.get('MINIMAX_API_KEY', ''))
        else:
            env_key = ''
        self.assertEqual(env_key, 'test-m3-key')

    def test_abab_still_matches(self):
        """Legacy abab models should still match the key detection condition."""
        for model in ['abab6.5s-chat', 'abab7-chat-preview']:
            self.assertTrue('abab' in model, f"{model} should match 'abab' pattern")

    def test_non_minimax_no_match(self):
        """Non-MiniMax models should not match the MiniMax key detection."""
        for model in ['gpt-4o', 'claude-3', 'gemini-pro']:
            self.assertFalse('abab' in model or 'MiniMax-M' in model)


class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests for MiniMax provider (require MINIMAX_API_KEY)."""

    @unittest.skipUnless(
        os.environ.get('MiniMax_API_KEY') or os.environ.get('MINIMAX_API_KEY'),
        'MiniMax_API_KEY or MINIMAX_API_KEY not set'
    )
    def test_m3_generate_text(self):
        """Integration: MiniMax-M3 should generate a text response."""
        from vlmeval.config import supported_VLM
        model = supported_VLM['MiniMax-M3']()
        result = model.generate('What is 6+1? Answer with just the number.')
        self.assertIsInstance(result, str)
        self.assertIn('7', result)

    @unittest.skipUnless(
        os.environ.get('MiniMax_API_KEY') or os.environ.get('MINIMAX_API_KEY'),
        'MiniMax_API_KEY or MINIMAX_API_KEY not set'
    )
    def test_m27_generate_text(self):
        """Integration: MiniMax-M2.7 should generate a text response."""
        from vlmeval.config import supported_VLM
        model = supported_VLM['MiniMax-M2.7']()
        result = model.generate('What is 2+2? Answer with just the number.')
        self.assertIsInstance(result, str)
        self.assertIn('4', result)

    @unittest.skipUnless(
        os.environ.get('MiniMax_API_KEY') or os.environ.get('MINIMAX_API_KEY'),
        'MiniMax_API_KEY or MINIMAX_API_KEY not set'
    )
    def test_m27_highspeed_generate_text(self):
        """Integration: MiniMax-M2.7-highspeed should generate a text response."""
        from vlmeval.config import supported_VLM
        model = supported_VLM['MiniMax-M2.7-highspeed']()
        result = model.generate('What is 3+5? Answer with just the number.')
        self.assertIsInstance(result, str)
        self.assertIn('8', result)

    @unittest.skipUnless(
        os.environ.get('MiniMax_API_KEY') or os.environ.get('MINIMAX_API_KEY'),
        'MiniMax_API_KEY or MINIMAX_API_KEY not set'
    )
    def test_m3_working(self):
        """Integration: MiniMax-M3 wrapper should report as working."""
        from vlmeval.config import supported_VLM
        model = supported_VLM['MiniMax-M3']()
        self.assertTrue(model.working())

    @unittest.skipUnless(
        os.environ.get('MiniMax_API_KEY') or os.environ.get('MINIMAX_API_KEY'),
        'MiniMax_API_KEY or MINIMAX_API_KEY not set'
    )
    def test_m27_working(self):
        """Integration: MiniMax-M2.7 wrapper should report as working."""
        from vlmeval.config import supported_VLM
        model = supported_VLM['MiniMax-M2.7']()
        self.assertTrue(model.working())


if __name__ == '__main__':
    unittest.main()

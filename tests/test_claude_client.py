"""
Unit tests for core/claude_client.py.

Run with:
    cd /Users/turje87/Desktop/TradeBots && python -m pytest tests/test_claude_client.py -v
"""

import time
import unittest
from unittest.mock import MagicMock, patch, call

import anthropic
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_API_KEY = "sk-ant-test-key"
FAKE_RESPONSE_TEXT = "BUY signal detected with high confidence."
FAKE_MODEL = "claude-test-model"


def _make_mock_response(text: str = FAKE_RESPONSE_TEXT) -> MagicMock:
    """Build a mock that looks like an anthropic.Message with one TextBlock."""
    mock_content = MagicMock()
    mock_content.text = text
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    return mock_response


def _make_client(agent_name: str = "test_agent"):
    """Import and instantiate ClaudeClient with a patched API key."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": FAKE_API_KEY}):
        # Re-import settings so ANTHROPIC_API_KEY picks up the patched value
        import importlib
        import config.settings as settings_mod
        importlib.reload(settings_mod)

        import core.claude_client as cc_mod
        importlib.reload(cc_mod)

        from core.claude_client import ClaudeClient
        return ClaudeClient(agent_name=agent_name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClaudeClientInit:
    def test_raises_environment_error_when_key_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            import importlib
            import config.settings as settings_mod
            importlib.reload(settings_mod)
            import core.claude_client as cc_mod
            importlib.reload(cc_mod)
            from core.claude_client import ClaudeClient
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                ClaudeClient(agent_name="scanner")

    def test_logger_name_uses_agent_name(self):
        client = _make_client(agent_name="orchestrator")
        assert client._logger.name == "claude_client.orchestrator"

    def test_agent_name_stored(self):
        client = _make_client(agent_name="strategy.technical")
        assert client.agent_name == "strategy.technical"


class TestClaudeClientSuccessfulCall:
    def test_returns_response_text(self):
        client = _make_client()
        mock_response = _make_mock_response(FAKE_RESPONSE_TEXT)
        with patch.object(client._client.messages, "create", return_value=mock_response):
            result = client.call(
                system_prompt="You are a trading bot.",
                user_message="Analyse TSLA",
            )
        assert result == FAKE_RESPONSE_TEXT

    def test_passes_correct_args_to_sdk(self):
        client = _make_client()
        mock_response = _make_mock_response()
        with patch.object(client._client.messages, "create", return_value=mock_response) as mock_create:
            client.call(
                system_prompt="System",
                user_message="User message",
                model=FAKE_MODEL,
                max_tokens=512,
                temperature=0.1,
            )
        mock_create.assert_called_once_with(
            model=FAKE_MODEL,
            max_tokens=512,
            system="System",
            messages=[{"role": "user", "content": "User message"}],
            temperature=0.1,
        )

    def test_uses_default_model_when_none(self):
        client = _make_client()
        mock_response = _make_mock_response()
        with patch.object(client._client.messages, "create", return_value=mock_response) as mock_create:
            client.call(system_prompt="s", user_message="u")
        # model kwarg must not be None
        called_model = mock_create.call_args.kwargs.get("model") or mock_create.call_args[1].get("model")
        assert called_model is not None
        assert called_model != "None"

    def test_single_attempt_on_success(self):
        client = _make_client()
        mock_response = _make_mock_response()
        with patch.object(client._client.messages, "create", return_value=mock_response) as mock_create:
            client.call(system_prompt="s", user_message="u")
        assert mock_create.call_count == 1


class TestClaudeClientLogging:
    def test_logs_success_info_message(self):
        client = _make_client(agent_name="scanner")
        mock_response = _make_mock_response(FAKE_RESPONSE_TEXT)
        with patch.object(client._client.messages, "create", return_value=mock_response):
            with patch.object(client._logger, "info") as mock_log:
                client.call(system_prompt="s", user_message="Analyse TSLA now")
        assert mock_log.call_count == 1
        log_msg = mock_log.call_args[0][0]
        assert "Claude call" in log_msg
        assert "agent=%s" in log_msg  # uses lazy % formatting

    def test_log_contains_agent_name_as_arg(self):
        client = _make_client(agent_name="watcher")
        mock_response = _make_mock_response()
        with patch.object(client._client.messages, "create", return_value=mock_response):
            with patch.object(client._logger, "info") as mock_log:
                client.call(system_prompt="s", user_message="u")
        log_args = mock_log.call_args[0]
        # Second positional arg after format string is agent_name
        assert "watcher" in log_args

    def test_log_includes_latency(self):
        client = _make_client()
        mock_response = _make_mock_response()
        with patch.object(client._client.messages, "create", return_value=mock_response):
            with patch.object(client._logger, "info") as mock_log:
                client.call(system_prompt="s", user_message="u")
        log_format = mock_log.call_args[0][0]
        assert "latency" in log_format

    def test_log_includes_input_and_output_preview(self):
        client = _make_client()
        long_input = "A" * 200
        long_output = "B" * 200
        mock_response = _make_mock_response(long_output)
        with patch.object(client._client.messages, "create", return_value=mock_response):
            with patch.object(client._logger, "info") as mock_log:
                client.call(system_prompt="s", user_message=long_input)
        log_args = mock_log.call_args[0]
        # The 100-char previews are passed as positional format args
        input_preview_arg = log_args[4]   # 5th positional arg
        output_preview_arg = log_args[5]  # 6th positional arg
        assert input_preview_arg == "A" * 100
        assert output_preview_arg == "B" * 100


class TestClaudeClientRetryOnRateLimitError:
    def test_retries_three_times_on_rate_limit_error(self):
        client = _make_client()
        rate_limit_exc = anthropic.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        with patch.object(
            client._client.messages, "create", side_effect=rate_limit_exc
        ) as mock_create:
            with patch("time.sleep"):  # skip actual sleeping
                with pytest.raises(anthropic.RateLimitError):
                    client.call(system_prompt="s", user_message="u")
        assert mock_create.call_count == 3

    def test_succeeds_on_second_attempt_after_rate_limit(self):
        client = _make_client()
        rate_limit_exc = anthropic.RateLimitError(
            message="Rate limit",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        mock_response = _make_mock_response(FAKE_RESPONSE_TEXT)
        with patch.object(
            client._client.messages,
            "create",
            side_effect=[rate_limit_exc, mock_response],
        ) as mock_create:
            with patch("time.sleep"):
                result = client.call(system_prompt="s", user_message="u")
        assert result == FAKE_RESPONSE_TEXT
        assert mock_create.call_count == 2

    def test_logs_retry_warning_on_each_failed_attempt(self):
        client = _make_client()
        rate_limit_exc = anthropic.RateLimitError(
            message="Rate limit",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        with patch.object(client._client.messages, "create", side_effect=rate_limit_exc):
            with patch("time.sleep"):
                with patch.object(client._logger, "warning") as mock_warn:
                    with pytest.raises(anthropic.RateLimitError):
                        client.call(system_prompt="s", user_message="u")
        assert mock_warn.call_count == 3

    def test_retries_on_api_connection_error(self):
        client = _make_client()
        conn_exc = anthropic.APIConnectionError(request=MagicMock())
        with patch.object(
            client._client.messages, "create", side_effect=conn_exc
        ) as mock_create:
            with patch("time.sleep"):
                with pytest.raises(anthropic.APIConnectionError):
                    client.call(system_prompt="s", user_message="u")
        assert mock_create.call_count == 3

    def test_retries_on_internal_server_error(self):
        client = _make_client()
        server_exc = anthropic.InternalServerError(
            message="Internal server error",
            response=MagicMock(status_code=500, headers={}),
            body={},
        )
        with patch.object(
            client._client.messages, "create", side_effect=server_exc
        ) as mock_create:
            with patch("time.sleep"):
                with pytest.raises(anthropic.InternalServerError):
                    client.call(system_prompt="s", user_message="u")
        assert mock_create.call_count == 3

    def test_exponential_backoff_sleep_values(self):
        """sleep() must be called with RETRY_BASE_DELAY * (2 ** attempt)."""
        from config.settings import RETRY_BASE_DELAY
        client = _make_client()
        rate_limit_exc = anthropic.RateLimitError(
            message="Rate limit",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        with patch.object(client._client.messages, "create", side_effect=rate_limit_exc):
            with patch("time.sleep") as mock_sleep:
                with pytest.raises(anthropic.RateLimitError):
                    client.call(system_prompt="s", user_message="u")
        # With 3 retries, sleep is called for attempts 0 and 1 (not after the last attempt)
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0] == call(RETRY_BASE_DELAY * (2 ** 0))
        assert mock_sleep.call_args_list[1] == call(RETRY_BASE_DELAY * (2 ** 1))


class TestClaudeClientNoRetryOnFatalErrors:
    def test_no_retry_on_authentication_error(self):
        client = _make_client()
        auth_exc = anthropic.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401, headers={}),
            body={},
        )
        with patch.object(
            client._client.messages, "create", side_effect=auth_exc
        ) as mock_create:
            with pytest.raises(anthropic.AuthenticationError):
                client.call(system_prompt="s", user_message="u")
        # Must fail immediately on the first attempt
        assert mock_create.call_count == 1

    def test_no_retry_on_bad_request_error(self):
        client = _make_client()
        bad_req_exc = anthropic.BadRequestError(
            message="Invalid request",
            response=MagicMock(status_code=400, headers={}),
            body={},
        )
        with patch.object(
            client._client.messages, "create", side_effect=bad_req_exc
        ) as mock_create:
            with pytest.raises(anthropic.BadRequestError):
                client.call(system_prompt="s", user_message="u")
        assert mock_create.call_count == 1

    def test_no_warning_log_on_fatal_error(self):
        client = _make_client()
        auth_exc = anthropic.AuthenticationError(
            message="Unauthorized",
            response=MagicMock(status_code=401, headers={}),
            body={},
        )
        with patch.object(client._client.messages, "create", side_effect=auth_exc):
            with patch.object(client._logger, "warning") as mock_warn:
                with pytest.raises(anthropic.AuthenticationError):
                    client.call(system_prompt="s", user_message="u")
        assert mock_warn.call_count == 0


class TestClaudeClientExhaustRetries:
    def test_raises_last_exception_after_exhausting_retries(self):
        client = _make_client()
        rate_limit_exc = anthropic.RateLimitError(
            message="Persistent rate limit",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        with patch.object(client._client.messages, "create", side_effect=rate_limit_exc):
            with patch("time.sleep"):
                with pytest.raises(anthropic.RateLimitError) as exc_info:
                    client.call(system_prompt="s", user_message="u")
        assert exc_info.value is rate_limit_exc

    def test_no_info_log_when_all_retries_fail(self):
        """No success info log should be emitted if the call ultimately fails."""
        client = _make_client()
        rate_limit_exc = anthropic.RateLimitError(
            message="Rate limit",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        with patch.object(client._client.messages, "create", side_effect=rate_limit_exc):
            with patch("time.sleep"):
                with patch.object(client._logger, "info") as mock_info:
                    with pytest.raises(anthropic.RateLimitError):
                        client.call(system_prompt="s", user_message="u")
        assert mock_info.call_count == 0

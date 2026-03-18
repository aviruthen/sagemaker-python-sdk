"""Tests for Pipeline role_arn validation."""
import pytest


def test_string_role_arn_accepted():
    """String role_arn should be accepted."""
    from sagemaker.workflow.pipeline import _validate_role_arn
    _validate_role_arn("arn:aws:iam::123:role/MyRole")  # Should not raise


def test_none_role_arn_accepted():
    """None role_arn should be accepted."""
    from sagemaker.workflow.pipeline import _validate_role_arn
    _validate_role_arn(None)  # Should not raise


def test_dict_role_arn_raises():
    """Dict role_arn should raise ValueError."""
    from sagemaker.workflow.pipeline import _validate_role_arn
    with pytest.raises(ValueError, match='role_arn must be a string'):
        _validate_role_arn({"key": "value"})


def test_int_role_arn_raises():
    """Int role_arn should raise ValueError."""
    from sagemaker.workflow.pipeline import _validate_role_arn
    with pytest.raises(ValueError, match='role_arn must be a string'):
        _validate_role_arn(12345)

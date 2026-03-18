"""Unit tests for Pipeline.upsert() fix."""
import pytest
from sagemaker.workflow.pipeline import upsert


def test_upsert_with_string_role_arn():
    result = upsert({"steps": []}, "arn:aws:iam::123:role/MyRole")
    assert result[1] == "arn:aws:iam::123:role/MyRole"


def test_upsert_with_dict_role_arn():
    result = upsert({"steps": []}, {"arn": "arn:aws:iam::123:role/MyRole"})
    assert result[1] == "arn:aws:iam::123:role/MyRole"


def test_upsert_with_invalid_role_arn_raises():
    with pytest.raises(TypeError):
        upsert({"steps": []}, 12345)

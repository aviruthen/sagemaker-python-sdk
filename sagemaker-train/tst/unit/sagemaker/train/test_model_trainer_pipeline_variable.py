# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Tests for PipelineVariable support in ModelTrainer."""
from __future__ import absolute_import

import pytest
from unittest.mock import MagicMock, patch

from sagemaker.core.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.core.helper.pipeline_variable import PipelineVariable
from sagemaker.train.utils import safe_serialize, _get_repo_name_from_image, _PIPELINE_VARIABLE_IMAGE_PLACEHOLDER


class TestSafeSerializeWithPipelineVariable:
    """Tests for safe_serialize handling of PipelineVariable objects."""

    def test_safe_serialize_string(self):
        """Test that plain strings are returned as-is."""
        assert safe_serialize("hello") == "hello"

    def test_safe_serialize_int(self):
        """Test that integers are JSON-serialized."""
        assert safe_serialize(5) == "5"

    def test_safe_serialize_float(self):
        """Test that floats are JSON-serialized."""
        assert safe_serialize(3.14) == "3.14"

    def test_safe_serialize_dict(self):
        """Test that dicts are JSON-serialized."""
        result = safe_serialize({"key": "value"})
        assert result == '{"key": "value"}'

    def test_safe_serialize_pipeline_variable_parameter_string(self):
        """Test that ParameterString is returned as the PipelineVariable object itself."""
        param = ParameterString(name="MyParam", default_value="test")
        result = safe_serialize(param)
        # Should return the PipelineVariable object, not raise TypeError
        assert isinstance(result, PipelineVariable)
        assert result is param

    def test_safe_serialize_pipeline_variable_parameter_integer(self):
        """Test that ParameterInteger is returned as the PipelineVariable object itself."""
        param = ParameterInteger(name="MaxDepth", default_value=5)
        result = safe_serialize(param)
        # Should return the PipelineVariable object, not raise TypeError
        assert isinstance(result, PipelineVariable)
        assert result is param


class TestGetRepoNameFromImage:
    """Tests for _get_repo_name_from_image handling of PipelineVariable objects."""

    def test_get_repo_name_from_image_string(self):
        """Test that a normal image URI returns the repo name."""
        image = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.0-1-cpu-py3"
        result = _get_repo_name_from_image(image)
        assert result == "sagemaker-xgboost"

    def test_get_repo_name_from_image_pipeline_variable(self):
        """Test that a PipelineVariable returns the placeholder constant."""
        param = ParameterString(name="TrainingImage", default_value="some-image")
        result = _get_repo_name_from_image(param)
        assert result == _PIPELINE_VARIABLE_IMAGE_PLACEHOLDER

    def test_get_repo_name_from_image_simple_string(self):
        """Test with a simple image name."""
        result = _get_repo_name_from_image("my-repo:latest")
        assert result == "my-repo"

    def test_get_repo_name_from_image_with_digest(self):
        """Test with an image URI containing a digest."""
        image = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-repo@sha256:abc123"
        result = _get_repo_name_from_image(image)
        assert result == "my-repo"


class TestModelTrainerValidationWithPipelineVariable:
    """Tests for ModelTrainer validation with PipelineVariable objects."""

    @patch("sagemaker.train.model_trainer.TrainDefaults")
    def test_training_image_accepts_parameter_string(self, mock_defaults):
        """Test that training_image accepts ParameterString."""
        from sagemaker.train.model_trainer import ModelTrainer
        from sagemaker.train.configs import Compute

        mock_session = MagicMock()
        mock_session.boto_region_name = "us-east-1"
        mock_session.default_bucket.return_value = "my-bucket"
        mock_session.default_bucket_prefix = None

        mock_defaults.get_sagemaker_session.return_value = mock_session
        mock_defaults.get_role.return_value = "arn:aws:iam::123456789012:role/SageMakerRole"
        mock_defaults.get_base_job_name.return_value = "test-job"
        mock_defaults.get_compute.return_value = Compute(
            instance_type="ml.m5.xlarge", instance_count=1
        )
        mock_defaults.get_stopping_condition.return_value = MagicMock()
        mock_defaults.get_output_data_config.return_value = MagicMock()

        param = ParameterString(name="TrainingImage", default_value="some-image-uri")

        # Should not raise
        trainer = ModelTrainer(
            training_image=param,
            compute=Compute(instance_type="ml.m5.xlarge", instance_count=1),
            sagemaker_session=mock_session,
            role="arn:aws:iam::123456789012:role/SageMakerRole",
        )
        assert trainer.training_image is param

    @patch("sagemaker.train.model_trainer.TrainDefaults")
    def test_algorithm_name_accepts_parameter_string(self, mock_defaults):
        """Test that algorithm_name accepts ParameterString."""
        from sagemaker.train.model_trainer import ModelTrainer
        from sagemaker.train.configs import Compute

        mock_session = MagicMock()
        mock_session.boto_region_name = "us-east-1"
        mock_session.default_bucket.return_value = "my-bucket"
        mock_session.default_bucket_prefix = None

        mock_defaults.get_sagemaker_session.return_value = mock_session
        mock_defaults.get_role.return_value = "arn:aws:iam::123456789012:role/SageMakerRole"
        mock_defaults.get_base_job_name.return_value = "test-job"
        mock_defaults.get_compute.return_value = Compute(
            instance_type="ml.m5.xlarge", instance_count=1
        )
        mock_defaults.get_stopping_condition.return_value = MagicMock()
        mock_defaults.get_output_data_config.return_value = MagicMock()

        param = ParameterString(name="AlgorithmName", default_value="some-algo")

        # Should not raise
        trainer = ModelTrainer(
            algorithm_name=param,
            compute=Compute(instance_type="ml.m5.xlarge", instance_count=1),
            sagemaker_session=mock_session,
            role="arn:aws:iam::123456789012:role/SageMakerRole",
        )
        assert trainer.algorithm_name is param

    @patch("sagemaker.train.model_trainer.TrainDefaults")
    def test_environment_values_accept_parameter_string(self, mock_defaults):
        """Test that environment dict values accept ParameterString."""
        from sagemaker.train.model_trainer import ModelTrainer
        from sagemaker.train.configs import Compute

        mock_session = MagicMock()
        mock_session.boto_region_name = "us-east-1"
        mock_session.default_bucket.return_value = "my-bucket"
        mock_session.default_bucket_prefix = None

        mock_defaults.get_sagemaker_session.return_value = mock_session
        mock_defaults.get_role.return_value = "arn:aws:iam::123456789012:role/SageMakerRole"
        mock_defaults.get_base_job_name.return_value = "test-job"
        mock_defaults.get_compute.return_value = Compute(
            instance_type="ml.m5.xlarge", instance_count=1
        )
        mock_defaults.get_stopping_condition.return_value = MagicMock()
        mock_defaults.get_output_data_config.return_value = MagicMock()

        env_param = ParameterString(name="EnvValue", default_value="val")

        # Should not raise
        trainer = ModelTrainer(
            training_image="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.0-1-cpu-py3",
            compute=Compute(instance_type="ml.m5.xlarge", instance_count=1),
            sagemaker_session=mock_session,
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            environment={"MY_VAR": env_param},
        )
        assert trainer.environment["MY_VAR"] is env_param

    @patch("sagemaker.train.model_trainer.TrainDefaults")
    def test_plain_string_values_still_work(self, mock_defaults):
        """Regression test: plain string values continue to work."""
        from sagemaker.train.model_trainer import ModelTrainer
        from sagemaker.train.configs import Compute

        mock_session = MagicMock()
        mock_session.boto_region_name = "us-east-1"
        mock_session.default_bucket.return_value = "my-bucket"
        mock_session.default_bucket_prefix = None

        mock_defaults.get_sagemaker_session.return_value = mock_session
        mock_defaults.get_role.return_value = "arn:aws:iam::123456789012:role/SageMakerRole"
        mock_defaults.get_base_job_name.return_value = "test-job"
        mock_defaults.get_compute.return_value = Compute(
            instance_type="ml.m5.xlarge", instance_count=1
        )
        mock_defaults.get_stopping_condition.return_value = MagicMock()
        mock_defaults.get_output_data_config.return_value = MagicMock()

        # Should not raise
        trainer = ModelTrainer(
            training_image="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.0-1-cpu-py3",
            compute=Compute(instance_type="ml.m5.xlarge", instance_count=1),
            sagemaker_session=mock_session,
            role="arn:aws:iam::123456789012:role/SageMakerRole",
        )
        assert trainer.training_image == "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.0-1-cpu-py3"

    def test_validation_rejects_no_image_or_algorithm(self):
        """Test that validation rejects when neither training_image nor algorithm_name is provided."""
        from sagemaker.train.model_trainer import ModelTrainer

        trainer = ModelTrainer.__new__(ModelTrainer)
        with pytest.raises(ValueError, match="Atleast one of"):
            trainer._validate_training_image_and_algorithm_name(None, None)

    def test_validation_rejects_both_image_and_algorithm(self):
        """Test that validation rejects when both training_image and algorithm_name are provided."""
        from sagemaker.train.model_trainer import ModelTrainer

        trainer = ModelTrainer.__new__(ModelTrainer)
        with pytest.raises(ValueError, match="Only one of"):
            trainer._validate_training_image_and_algorithm_name("image", "algo")

    def test_validation_rejects_both_pipeline_variables(self):
        """Test that validation rejects when both are PipelineVariables."""
        from sagemaker.train.model_trainer import ModelTrainer

        trainer = ModelTrainer.__new__(ModelTrainer)
        img_param = ParameterString(name="Image", default_value="img")
        algo_param = ParameterString(name="Algo", default_value="algo")
        with pytest.raises(ValueError, match="Only one of"):
            trainer._validate_training_image_and_algorithm_name(img_param, algo_param)

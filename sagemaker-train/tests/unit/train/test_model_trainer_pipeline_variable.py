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
"""Tests for PipelineVariable support in ModelTrainer.

Verifies that ModelTrainer fields accept PipelineVariable objects
(e.g., ParameterString) in addition to their concrete types, following
the existing V3 pattern established by SourceCode and OutputDataConfig.

Also verifies that safe_serialize correctly handles PipelineVariable objects
in hyperparameters (returning them as-is instead of attempting json.dumps),
and that _create_training_job_args preserves PipelineVariable objects through
the serialization pipeline.
"""
from __future__ import absolute_import

import pytest
from pydantic import ValidationError
from unittest.mock import patch, MagicMock

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.helper.pipeline_variable import PipelineVariable, StrPipeVar
from sagemaker.core.workflow.parameters import (
    ParameterString,
    ParameterInteger,
    ParameterFloat,
)
from sagemaker.train.model_trainer import ModelTrainer, Mode
from sagemaker.train.configs import (
    Compute,
    StoppingCondition,
    OutputDataConfig,
)
from sagemaker.train.utils import safe_serialize
from sagemaker.train.defaults import DEFAULT_INSTANCE_TYPE


DEFAULT_IMAGE = "000000000000.dkr.ecr.us-west-2.amazonaws.com/dummy-image:latest"
DEFAULT_BUCKET = "sagemaker-us-west-2-000000000000"
DEFAULT_ROLE = "arn:aws:iam::000000000000:role/test-role"
DEFAULT_BUCKET_PREFIX = "sample-prefix"
DEFAULT_REGION = "us-west-2"
DEFAULT_COMPUTE = Compute(instance_type=DEFAULT_INSTANCE_TYPE, instance_count=1)
DEFAULT_STOPPING = StoppingCondition(max_runtime_in_seconds=3600)
DEFAULT_OUTPUT = OutputDataConfig(
    s3_output_path=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BUCKET_PREFIX}/test-job",
)


@pytest.fixture(scope="module", autouse=True)
def modules_session():
    with patch("sagemaker.train.Session", spec=Session) as session_mock:
        session_instance = session_mock.return_value
        session_instance.default_bucket.return_value = DEFAULT_BUCKET
        session_instance.get_caller_identity_arn.return_value = DEFAULT_ROLE
        session_instance.default_bucket_prefix = DEFAULT_BUCKET_PREFIX
        session_instance.boto_session = MagicMock(spec="boto3.session.Session")
        session_instance.boto_region_name = DEFAULT_REGION
        yield session_instance


class TestModelTrainerPipelineVariableAcceptance:
    """Test that ModelTrainer fields accept PipelineVariable objects."""

    def test_training_image_accepts_parameter_string(self):
        """ModelTrainer.training_image should accept ParameterString (GH#5524)."""
        param = ParameterString(name="TrainingImage", default_value=DEFAULT_IMAGE)
        trainer = ModelTrainer(
            training_image=param,
            base_job_name="pipeline-test-job",  # Required: PipelineVariable can't generate job name
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.training_image is param

    def test_algorithm_name_accepts_parameter_string(self):
        """ModelTrainer.algorithm_name should accept ParameterString."""
        param = ParameterString(name="AlgorithmName", default_value="my-algo-arn")
        trainer = ModelTrainer(
            algorithm_name=param,
            base_job_name="pipeline-test-job",  # Required: PipelineVariable can't generate job name
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.algorithm_name is param

    def test_training_input_mode_accepts_parameter_string(self):
        """ModelTrainer.training_input_mode should accept ParameterString."""
        param = ParameterString(name="InputMode", default_value="File")
        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            training_input_mode=param,
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.training_input_mode is param

    def test_environment_values_accept_parameter_string(self):
        """ModelTrainer.environment dict values should accept ParameterString."""
        param = ParameterString(name="DatasetVersion", default_value="v1")
        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            environment={"DATASET_VERSION": param, "STATIC_VAR": "hello"},
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.environment["DATASET_VERSION"] is param
        assert trainer.environment["STATIC_VAR"] == "hello"


class TestModelTrainerRealValuesStillWork:
    """Regression tests: verify that passing real values still works after the change."""

    def test_training_image_accepts_real_string(self):
        """ModelTrainer.training_image should still accept a plain string."""
        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.training_image == DEFAULT_IMAGE

    def test_algorithm_name_accepts_real_string(self):
        """ModelTrainer.algorithm_name should still accept a plain string."""
        trainer = ModelTrainer(
            algorithm_name="arn:aws:sagemaker:us-west-2:000000000000:algorithm/my-algo",
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.algorithm_name == "arn:aws:sagemaker:us-west-2:000000000000:algorithm/my-algo"

    def test_training_input_mode_accepts_real_string(self):
        """ModelTrainer.training_input_mode should still accept a plain string."""
        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            training_input_mode="Pipe",
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.training_input_mode == "Pipe"

    def test_environment_accepts_real_string_values(self):
        """ModelTrainer.environment should still accept plain string values."""
        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            environment={"KEY1": "value1", "KEY2": "value2"},
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.environment == {"KEY1": "value1", "KEY2": "value2"}

    def test_training_image_rejects_invalid_type(self):
        """ModelTrainer.training_image should still reject invalid types (e.g., int)."""
        with pytest.raises(ValidationError):
            ModelTrainer(
                training_image=12345,
                role=DEFAULT_ROLE,
                compute=DEFAULT_COMPUTE,
                stopping_condition=DEFAULT_STOPPING,
                output_data_config=DEFAULT_OUTPUT,
            )


class TestSafeSerializeWithPipelineVariables:
    """Tests that safe_serialize handles PipelineVariable objects correctly.

    The safe_serialize function must return PipelineVariable objects as-is
    instead of attempting json.dumps(), which would raise TypeError.
    """

    def test_safe_serialize_with_parameter_integer_returns_pipeline_variable(self):
        """safe_serialize should return ParameterInteger as-is."""
        param = ParameterInteger(name="MaxDepth", default_value=5)
        result = safe_serialize(param)
        assert result is param
        assert isinstance(result, PipelineVariable)

    def test_safe_serialize_with_parameter_string_returns_pipeline_variable(self):
        """safe_serialize should return ParameterString as-is."""
        param = ParameterString(name="Optimizer", default_value="adam")
        result = safe_serialize(param)
        assert result is param
        assert isinstance(result, PipelineVariable)

    def test_safe_serialize_with_parameter_float_returns_pipeline_variable(self):
        """safe_serialize should return ParameterFloat as-is."""
        param = ParameterFloat(name="LearningRate", default_value=0.01)
        result = safe_serialize(param)
        assert result is param
        assert isinstance(result, PipelineVariable)

    def test_safe_serialize_still_handles_strings(self):
        """safe_serialize should return plain strings as-is (no quotes wrapping)."""
        result = safe_serialize("hello")
        assert result == "hello"

    def test_safe_serialize_still_handles_integers(self):
        """safe_serialize should JSON-encode integers."""
        result = safe_serialize(42)
        assert result == "42"

    def test_safe_serialize_still_handles_dicts(self):
        """safe_serialize should JSON-encode dicts."""
        result = safe_serialize({"key": "value"})
        assert result == '{"key": "value"}'

    def test_safe_serialize_still_handles_floats(self):
        """safe_serialize should JSON-encode floats."""
        result = safe_serialize(0.01)
        assert result == "0.01"

    def test_safe_serialize_still_handles_booleans(self):
        """safe_serialize should JSON-encode booleans."""
        assert safe_serialize(True) == "true"
        assert safe_serialize(False) == "false"


class TestModelTrainerHyperparametersWithPipelineVariables:
    """Tests that ModelTrainer accepts PipelineVariable objects in hyperparameters."""

    def test_hyperparameters_accept_pipeline_variable_values(self):
        """ModelTrainer should accept PipelineVariable objects as hyperparameter values."""
        max_depth = ParameterInteger(name="MaxDepth", default_value=5)
        learning_rate = ParameterFloat(name="LearningRate", default_value=0.01)
        optimizer = ParameterString(name="Optimizer", default_value="adam")

        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
            hyperparameters={
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "static_param": 10,
            },
        )
        assert trainer.hyperparameters["max_depth"] is max_depth
        assert trainer.hyperparameters["learning_rate"] is learning_rate
        assert trainer.hyperparameters["optimizer"] is optimizer
        assert trainer.hyperparameters["static_param"] == 10

    def test_create_training_job_args_with_pipeline_variable_hyperparameters(self):
        """_create_training_job_args should preserve PipelineVariable in hyper_parameters."""
        max_depth = ParameterInteger(name="MaxDepth", default_value=5)
        learning_rate = ParameterFloat(name="LearningRate", default_value=0.01)

        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
            hyperparameters={
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "epochs": 10,
                "verbose": "true",
            },
        )

        training_args = trainer._create_training_job_args()
        hyper_params = training_args["hyper_parameters"]

        # PipelineVariable objects should be preserved as-is by safe_serialize
        assert hyper_params["max_depth"] is max_depth
        assert hyper_params["learning_rate"] is learning_rate
        # Regular values should be serialized to strings
        assert hyper_params["epochs"] == "10"
        assert hyper_params["verbose"] == "true"

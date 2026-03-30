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

Verify that ModelTrainer fields accept PipelineVariable objects
(e.g., ParameterString) in addition to their concrete types, following
the existing V3 pattern established by SourceCode and OutputDataConfig.

Also verify that safe_serialize correctly handles PipelineVariable objects
in hyperparameters (returning them as-is instead of attempting json.dumps),
and that _create_training_job_args preserves PipelineVariable objects through
the serialization pipeline.

See: https://github.com/aws/sagemaker-python-sdk/issues/5504
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


@pytest.fixture(scope="module")
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
        """Verify ModelTrainer.training_image accepts ParameterString (GH#5504)."""
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
        """Verify ModelTrainer.algorithm_name accepts ParameterString."""
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
        """Verify ModelTrainer.training_input_mode accepts ParameterString."""
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
        """Verify ModelTrainer.environment dict values accept ParameterString."""
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
        """Verify ModelTrainer.training_image still accepts a plain string."""
        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.training_image == DEFAULT_IMAGE

    def test_algorithm_name_accepts_real_string(self):
        """Verify ModelTrainer.algorithm_name still accepts a plain string."""
        trainer = ModelTrainer(
            algorithm_name="arn:aws:sagemaker:us-west-2:000000000000:algorithm/my-algo",
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
        )
        assert trainer.algorithm_name == "arn:aws:sagemaker:us-west-2:000000000000:algorithm/my-algo"

    def test_training_input_mode_accepts_real_string(self):
        """Verify ModelTrainer.training_input_mode still accepts a plain string."""
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
        """Verify ModelTrainer.environment still accepts plain string values."""
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
        """Verify ModelTrainer.training_image still rejects invalid types (e.g., int)."""
        with pytest.raises(ValidationError):
            ModelTrainer(
                training_image=12345,
                role=DEFAULT_ROLE,
                compute=DEFAULT_COMPUTE,
                stopping_condition=DEFAULT_STOPPING,
                output_data_config=DEFAULT_OUTPUT,
            )


class TestSafeSerializeWithPipelineVariables:
    """Verify that safe_serialize handles PipelineVariable objects correctly.

    The safe_serialize function must return PipelineVariable objects as-is
    instead of attempting json.dumps(), which would raise TypeError.
    See: https://github.com/aws/sagemaker-python-sdk/issues/5504
    """

    @pytest.mark.parametrize("param", [
        ParameterInteger(name="MaxDepth", default_value=5),
        ParameterString(name="Optimizer", default_value="adam"),
        ParameterFloat(name="LearningRate", default_value=0.01),
    ])
    def test_safe_serialize_returns_pipeline_variable_as_is(self, param):
        """Verify safe_serialize returns PipelineVariable objects as-is."""
        result = safe_serialize(param)
        assert result is param
        assert isinstance(result, PipelineVariable)

    @pytest.mark.parametrize("input_val,expected", [
        ("hello", "hello"),
        (42, "42"),
        ({"key": "value"}, '{"key": "value"}'),
        (0.01, "0.01"),
        (True, "true"),
        (False, "false"),
    ])
    def test_safe_serialize_handles_normal_types(self, input_val, expected):
        """Verify safe_serialize correctly serializes normal (non-PipelineVariable) types."""
        result = safe_serialize(input_val)
        assert result == expected


class TestModelTrainerHyperparametersWithPipelineVariables:
    """Verify that ModelTrainer accepts PipelineVariable objects in hyperparameters.

    See: https://github.com/aws/sagemaker-python-sdk/issues/5504
    """

    def test_hyperparameters_accept_pipeline_variable_values(self):
        """Verify ModelTrainer accepts PipelineVariable objects as hyperparameter values."""
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

    def test_create_training_job_args_with_pipeline_variable_hyperparameters(
        self, modules_session
    ):
        """Verify _create_training_job_args preserves PipelineVariable in hyper_parameters."""
        max_depth = ParameterInteger(name="MaxDepth", default_value=5)
        learning_rate = ParameterFloat(name="LearningRate", default_value=0.01)

        trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            role=DEFAULT_ROLE,
            compute=DEFAULT_COMPUTE,
            stopping_condition=DEFAULT_STOPPING,
            output_data_config=DEFAULT_OUTPUT,
            sagemaker_session=modules_session,
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

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
"""Tests for ProcessingInput source parameter and local file upload behavior."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from sagemaker.core.shapes import ProcessingInput, ProcessingS3Input
from sagemaker.core.processing import Processor

# Test constants
FAKE_ACCOUNT_ID = "012345678901"
FAKE_ROLE_ARN = f"arn:aws:iam::{FAKE_ACCOUNT_ID}:role/SageMakerRole"
FAKE_IMAGE_URI = (
    f"{FAKE_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com/my-image:latest"
)
FAKE_BUCKET = "my-bucket"


@pytest.fixture
def sagemaker_session():
    session = MagicMock()
    session.default_bucket.return_value = FAKE_BUCKET
    session.default_bucket_prefix = None
    session.expand_role.return_value = FAKE_ROLE_ARN
    session.sagemaker_client = MagicMock()
    session.sagemaker_config = None
    type(session).local_mode = PropertyMock(return_value=False)
    return session


@pytest.fixture
def processor(sagemaker_session):
    return Processor(
        role=FAKE_ROLE_ARN,
        image_uri=FAKE_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=sagemaker_session,
    )


class TestProcessingInputSourceParameter:
    """Tests for the 'source' convenience parameter on ProcessingInput."""

    def test_source_parameter_creates_s3_input(self):
        """Test that providing 'source' auto-creates a ProcessingS3Input."""
        proc_input = ProcessingInput(
            input_name="my-input",
            source="/local/path/to/data",
        )
        assert proc_input.s3_input is not None
        assert proc_input.s3_input.s3_uri == "/local/path/to/data"
        assert proc_input.s3_input.s3_data_type == "S3Prefix"
        assert proc_input.s3_input.s3_input_mode == "File"
        assert proc_input.source == "/local/path/to/data"

    def test_source_with_s3_uri_passthrough(self):
        """Test that 'source' with an S3 URI creates s3_input."""
        proc_input = ProcessingInput(
            input_name="my-input",
            source="s3://my-bucket/my-prefix/data",
        )
        assert proc_input.s3_input is not None
        assert proc_input.s3_input.s3_uri == "s3://my-bucket/my-prefix/data"

    def test_source_and_s3_input_raises_error(self):
        """Test that providing both 'source' and 's3_input' raises ValueError."""
        with pytest.raises(
            ValueError, match="Cannot specify both 'source' and 's3_input'"
        ):
            ProcessingInput(
                input_name="my-input",
                source="/local/path/to/data",
                s3_input=ProcessingS3Input(
                    s3_uri="s3://my-bucket/data",
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                ),
            )

    def test_without_source_works_as_before(self):
        """Test that ProcessingInput without 'source' works as before."""
        proc_input = ProcessingInput(
            input_name="my-input",
            s3_input=ProcessingS3Input(
                s3_uri="s3://my-bucket/data",
                local_path="/opt/ml/processing/input",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
            ),
        )
        assert proc_input.s3_input.s3_uri == "s3://my-bucket/data"
        assert proc_input.source is None

    def test_source_none_and_s3_input_none_no_dataset(self):
        """Test ProcessingInput with neither source nor s3_input.

        When neither source nor s3_input is provided (and no
        dataset_definition), _normalize_inputs should raise ValueError.
        """
        proc_input = ProcessingInput(input_name="my-input")
        assert proc_input.s3_input is None or proc_input.s3_input.__class__.__name__ == "Unassigned"
        assert proc_input.source is None


class TestNormalizeInputsLocalUpload:
    """Tests for _normalize_inputs handling of local file paths."""

    @patch("sagemaker.core.processing.s3.S3Uploader.upload")
    def test_local_file_path_uploads_to_s3(
        self, mock_upload, processor
    ):
        """Test that a local file path in s3_uri triggers upload."""
        expected_uri = (
            f"s3://{FAKE_BUCKET}/job-name/input/my-input/data.csv"
        )
        mock_upload.return_value = expected_uri
        processor._current_job_name = "my-job"

        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False
        ) as f:
            local_path = f.name
            f.write(b"col1,col2\n1,2\n")

        try:
            inputs = [
                ProcessingInput(
                    input_name="my-input",
                    s3_input=ProcessingS3Input(
                        s3_uri=local_path,
                        s3_data_type="S3Prefix",
                        s3_input_mode="File",
                    ),
                )
            ]

            normalized = processor._normalize_inputs(inputs)

            assert len(normalized) == 1
            assert normalized[0].s3_input.s3_uri == expected_uri
            mock_upload.assert_called_once()
        finally:
            os.unlink(local_path)

    @patch("sagemaker.core.processing.s3.S3Uploader.upload")
    def test_source_local_path_uploads_to_s3(
        self, mock_upload, processor
    ):
        """Test that using 'source' with a local path triggers upload."""
        expected_uri = (
            f"s3://{FAKE_BUCKET}/job-name/input/my-input/data.csv"
        )
        mock_upload.return_value = expected_uri
        processor._current_job_name = "my-job"

        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False
        ) as f:
            local_path = f.name
            f.write(b"col1,col2\n1,2\n")

        try:
            inputs = [
                ProcessingInput(
                    input_name="my-input",
                    source=local_path,
                )
            ]

            normalized = processor._normalize_inputs(inputs)

            assert len(normalized) == 1
            assert normalized[0].s3_input.s3_uri == expected_uri
            mock_upload.assert_called_once()
        finally:
            os.unlink(local_path)

    def test_s3_uri_does_not_upload(self, processor):
        """Test that an S3 URI does not trigger upload."""
        processor._current_job_name = "my-job"

        inputs = [
            ProcessingInput(
                input_name="my-input",
                s3_input=ProcessingS3Input(
                    s3_uri="s3://my-bucket/existing-data",
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                ),
            )
        ]

        with patch(
            "sagemaker.core.processing.s3.S3Uploader.upload"
        ) as mock_upload:
            normalized = processor._normalize_inputs(inputs)

            assert len(normalized) == 1
            assert (
                normalized[0].s3_input.s3_uri
                == "s3://my-bucket/existing-data"
            )
            mock_upload.assert_not_called()

    @patch("sagemaker.core.processing.s3.S3Uploader.upload")
    def test_local_dir_path_uploads_to_s3(
        self, mock_upload, processor
    ):
        """Test that a local directory path triggers upload."""
        expected_uri = (
            f"s3://{FAKE_BUCKET}/job-name/input/my-input"
        )
        mock_upload.return_value = expected_uri
        processor._current_job_name = "my-job"

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, "data.csv"), "w"
            ) as f:
                f.write("col1,col2\n1,2\n")

            inputs = [
                ProcessingInput(
                    input_name="my-input",
                    source=tmpdir,
                )
            ]

            normalized = processor._normalize_inputs(inputs)

            assert len(normalized) == 1
            assert normalized[0].s3_input.s3_uri == expected_uri
            mock_upload.assert_called_once()

    def test_no_s3_input_no_source_no_dataset_raises_error(
        self, processor
    ):
        """Test that missing s3_input, source, and dataset raises error."""
        processor._current_job_name = "my-job"

        inputs = [
            ProcessingInput(input_name="my-input")
        ]

        with pytest.raises(ValueError, match="has no s3_input"):
            processor._normalize_inputs(inputs)

    @patch("sagemaker.core.processing.s3.S3Uploader.upload")
    def test_pipeline_config_generates_correct_s3_path(
        self, mock_upload, processor
    ):
        """Test that pipeline config generates the correct S3 path."""
        # _normalize_inputs imports _pipeline_config from
        # sagemaker.core.workflow.utilities at call time via:
        #   from sagemaker.core.workflow.utilities import _pipeline_config
        # So we patch it at the module where it's looked up.
        mock_config = MagicMock()
        mock_config.pipeline_name = "my-pipeline"
        mock_config.step_name = "my-step"

        expected_uri = (
            f"s3://{FAKE_BUCKET}/my-pipeline/my-step/input/my-input"
        )
        mock_upload.return_value = expected_uri
        processor._current_job_name = "my-job"

        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False
        ) as f:
            local_path = f.name
            f.write(b"col1,col2\n1,2\n")

        try:
            inputs = [
                ProcessingInput(
                    input_name="my-input",
                    source=local_path,
                )
            ]

            with patch(
                "sagemaker.core.workflow.utilities._pipeline_config",
                mock_config,
            ):
                normalized = processor._normalize_inputs(inputs)

            assert len(normalized) == 1
            mock_upload.assert_called_once()
            call_kwargs = mock_upload.call_args[1]
            assert "my-pipeline" in call_kwargs["desired_s3_uri"]
            assert "my-step" in call_kwargs["desired_s3_uri"]
        finally:
            os.unlink(local_path)

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0
# (the "License"). You may not use this file except in compliance
# with the License. A copy of the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file.

# Test comment: This module handles pipeline upsert operations. Added per reviewer request.

import logging

logger = logging.getLogger(__name__)


class Pipeline:
    def upsert(self, role_arn=None, **kwargs):
        """Create or update the pipeline.

        Args:
            role_arn (str): The ARN of the IAM role.
        """
        if role_arn is not None and not isinstance(role_arn, str):
            raise ValueError(
                f"role_arn must be a string, got: {type(role_arn).__name__}"
            )
        logger.info('Upserting pipeline with role_arn=%s', role_arn)

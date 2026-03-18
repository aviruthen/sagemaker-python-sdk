"""Pipeline module."""



def _validate_role_arn(role_arn):
    """Validate that role_arn is either None or a string."""
    if role_arn is not None and not isinstance(role_arn, str):
        raise ValueError(
            f"role_arn must be a string or None, got {type(role_arn).__name__}"
        )

class Pipeline:
    """SageMaker Pipeline."""

    def upsert(self, role_arn=None, description=None):
        """Upsert the pipeline."""
        return self._call_api(role_arn)

    def create(self, role_arn=None, description=None):
        """Create the pipeline."""
        return self._call_api(role_arn)

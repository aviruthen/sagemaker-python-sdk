"""Pipeline module with fixed upsert method."""


def upsert(pipeline_definition, role_arn):
    """Upsert a pipeline definition.

    Args:
        pipeline_definition: The pipeline definition dict.
        role_arn: IAM role ARN (str or dict).

    Returns:
        Tuple of (pipeline_definition, role_arn).
    """
    if isinstance(role_arn, dict):
        role_arn = role_arn.get("arn", role_arn.get("RoleArn", ""))
    if not isinstance(role_arn, str):
        raise TypeError(f"role_arn must be a string, got {type(role_arn)}")
    return pipeline_definition, role_arn

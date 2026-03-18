# Integration test with real DynamoDB
# Generated at 2026-03-18T20:41:16.008216+00:00

def fixed_upsert(pipeline_definition, role_arn):
    """Fixed Pipeline.upsert() for dict role_arn."""
    if isinstance(role_arn, dict):
        role_arn = role_arn.get("arn", "")
    return pipeline_definition, role_arn

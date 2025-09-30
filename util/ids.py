"""ID generation utilities."""
import uuid
import secrets


def generate_conn_id() -> str:
    """Generate a unique connection ID."""
    return f"conn_{secrets.token_hex(8)}"


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def generate_job_id() -> str:
    """Generate a unique job ID."""
    return f"job_{secrets.token_hex(6)}"

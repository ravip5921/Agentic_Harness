import pytest
from pydantic import ValidationError
from harness.models import TaskRequest


def test_generic_request():
    task = TaskRequest(task="Perform an approved repository change", parameters={"custom": [1, True]})
    assert task.max_dev_revisions == 2
    assert task.parameters["custom"] == [1, True]


def test_rejects_invalid_retry_limit():
    with pytest.raises(ValidationError):
        TaskRequest(task="Demo", max_dev_revisions=-1)


def test_rejects_extra_fields():
    with pytest.raises(ValidationError):
        TaskRequest(task="Demo", unknown_setting="x")

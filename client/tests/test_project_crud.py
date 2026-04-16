"""Project lifecycle: create -> get -> list -> delete."""
import pytest

import sys
from pathlib import Path
SDK_ROOT = Path(__file__).resolve().parent.parent
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from sdk import APIError, ProjectCreateRequest


@pytest.fixture()
def temp_project(client):
    """Create a throwaway project and tear it down after the test."""
    resp = client.create_project(ProjectCreateRequest(
        name="pytest-crud-temp",
        description="Throwaway project for CRUD test",
    ))
    pid = resp.project["project_id"]
    yield pid
    try:
        client.delete_project(pid)
    except APIError:
        pass


@pytest.mark.crud
class TestProjectCRUD:

    def test_list_projects_returns_list(self, client):
        resp = client.list_projects()
        assert isinstance(resp.projects, list)

    def test_get_project(self, client, project_id):
        resp = client.get_project(project_id)
        assert resp.project.get("project_id") or resp.project.get("name")

    def test_create_and_get(self, client, temp_project):
        resp = client.get_project(temp_project)
        assert resp.project["project_id"] == temp_project

    def test_list_includes_created_project(self, client, temp_project):
        resp = client.list_projects()
        ids = [p.get("project_id") or p.get("name") for p in resp.projects]
        assert temp_project in ids

    def test_delete_project(self, client):
        created = client.create_project(ProjectCreateRequest(
            name="pytest-delete-me",
        ))
        pid = created.project["project_id"]
        result = client.delete_project(pid)
        assert result.get("project_id") == pid

    def test_get_nonexistent_project(self, client):
        with pytest.raises(APIError) as exc_info:
            client.get_project("nonexistent-project-id-99999")
        assert exc_info.value.status_code in (403, 404)

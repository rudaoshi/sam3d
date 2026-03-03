import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--workspace_dir",
        action="store",
        required=True,
        help="Workspace directory for models"
    )

@pytest.fixture
def workspace_dir(request):
    return request.config.getoption("--workspace_dir")

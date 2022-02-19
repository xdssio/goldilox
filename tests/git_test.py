from goldilox.utils import get_git_info


def test_git_utils():
    git_info = get_git_info()
    assert git_info.get('branch')
    assert git_info.get('remote')

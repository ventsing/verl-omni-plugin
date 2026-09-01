"""
Tests for patch manager.
"""

import pytest

from shared.patch_manager import BasePatchManager, VersionChecker


class TestVersionChecker:
    """Tests for VersionChecker."""
    
    def test_check_version_ge(self):
        """Test >= version check."""
        # This test will only pass if the package is installed
        # For now, just test the logic
        result = VersionChecker.check("nonexistent_package", ">=1.0.0")
        assert result is False
    
    def test_check_all(self):
        """Test checking multiple versions."""
        requirements = {
            "nonexistent1": ">=1.0.0",
            "nonexistent2": ">=2.0.0",
        }
        result = VersionChecker.check_all(requirements)
        assert result is False


class TestBasePatchManager:
    """Tests for BasePatchManager."""
    
    def test_register_patch(self):
        """Test patch registration."""
        # Create a test patch manager
        class TestPatchManager(BasePatchManager):
            pass
        
        # Register a patch
        TestPatchManager.register_patch(
            name="test_patch",
            target_module="os",
            target_attr="getcwd",
            replacement_fn="os:getcwd",
            description="Test patch",
        )
        
        # Check that patch is registered
        assert "test_patch" in TestPatchManager.get_registered_patches()
    
    def test_get_registered_patches(self):
        """Test getting registered patches."""
        class TestPatchManager(BasePatchManager):
            pass
        
        # Register some patches
        TestPatchManager.register_patch(
            name="patch1",
            target_module="os",
            target_attr="getcwd",
            replacement_fn="os:getcwd",
        )
        
        TestPatchManager.register_patch(
            name="patch2",
            target_module="sys",
            target_attr="getcwd",
            replacement_fn="sys:getcwd",
        )
        
        # Check registered patches
        registered = TestPatchManager.get_registered_patches()
        assert "patch1" in registered
        assert "patch2" in registered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

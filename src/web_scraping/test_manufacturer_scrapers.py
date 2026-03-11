"""
Unit tests for manufacturer_scrapers.py
"""

import pytest
from unittest.mock import MagicMock, patch
from manufacturer_scrapers import (
    _normalize_mfr_name,
    check_robots_txt,
    _build_product_url,
    scrape_manufacturer_site,
    MANUFACTURER_REGISTRY,
    _robots_cache,
)


class TestNormalizeMfrName:
    """Test manufacturer name normalization with all 12+ names from CSV."""

    def test_skf(self):
        assert _normalize_mfr_name("SKF") == "skf"

    def test_nsk(self):
        assert _normalize_mfr_name("NSK") == "nsk"

    def test_timken(self):
        assert _normalize_mfr_name("Timken") == "timken"

    def test_dodge(self):
        assert _normalize_mfr_name("Dodge") == "dodge"

    def test_fag_schaeffler(self):
        assert _normalize_mfr_name("FAG (Schaeffler)") == "fag"

    def test_ina_schaeffler(self):
        assert _normalize_mfr_name("INA (Schaeffler)") == "ina"

    def test_leeson_electric(self):
        assert _normalize_mfr_name("LEESON ELECTRIC") == "leeson"

    def test_sealmaster(self):
        assert _normalize_mfr_name("SEALMASTER") == "sealmaster"

    def test_browning(self):
        assert _normalize_mfr_name("BROWNING") == "browning"

    def test_mcgill(self):
        assert _normalize_mfr_name("MCGILL") == "mcgill"

    def test_renold_jeffrey(self):
        assert _normalize_mfr_name("RENOLD JEFFREY") == "renold jeffrey"

    def test_rexnord_inc(self):
        assert _normalize_mfr_name("REXNORD INC") == "rexnord"

    def test_martin_sprocket(self):
        assert _normalize_mfr_name("MARTIN SPROCKET & GEAR CO") == "martin sprocket"

    def test_whitespace_handling(self):
        assert _normalize_mfr_name("  SKF  ") == "skf"

    def test_unknown_manufacturer(self):
        assert _normalize_mfr_name("ACME CORP") == "acme corp"


class TestCheckRobotsTxt:
    """Test robots.txt checking with mocked RobotFileParser."""

    def setup_method(self):
        # Clear cache between tests
        _robots_cache.clear()

    @patch("manufacturer_scrapers.RobotFileParser")
    def test_allowed_url(self, mock_rp_class):
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = True
        mock_rp_class.return_value = mock_rp

        assert check_robots_txt("https://www.skf.com/us/products?q=test") is True
        mock_rp.can_fetch.assert_called_once()

    @patch("manufacturer_scrapers.RobotFileParser")
    def test_disallowed_url(self, mock_rp_class):
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = False
        mock_rp_class.return_value = mock_rp

        assert check_robots_txt("https://www.skf.com/admin") is False

    @patch("manufacturer_scrapers.RobotFileParser")
    def test_fetch_error_returns_false(self, mock_rp_class):
        mock_rp = MagicMock()
        mock_rp.read.side_effect = Exception("Connection refused")
        mock_rp_class.return_value = mock_rp

        assert check_robots_txt("https://unreachable.example.com/page") is False

    @patch("manufacturer_scrapers.RobotFileParser")
    def test_caches_per_domain(self, mock_rp_class):
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = True
        mock_rp_class.return_value = mock_rp

        check_robots_txt("https://www.skf.com/page1")
        check_robots_txt("https://www.skf.com/page2")

        # RobotFileParser should only be created once for the same domain
        assert mock_rp_class.call_count == 1
        assert mock_rp.can_fetch.call_count == 2


class TestBuildProductUrl:
    """Test URL building with special characters in part numbers."""

    def test_simple_part_number(self):
        config = {"search_url_template": "https://example.com/search?q={part_number}"}
        url = _build_product_url(config, "21315")
        assert url == "https://example.com/search?q=21315"

    def test_part_number_with_comma(self):
        config = {"search_url_template": "https://example.com/search?q={part_number}"}
        url = _build_product_url(config, "IR10X13X12,5")
        assert "%2C" in url  # Comma should be encoded
        assert "," not in url.split("?q=")[1]

    def test_part_number_with_slash(self):
        config = {"search_url_template": "https://example.com/search?q={part_number}"}
        url = _build_product_url(config, "LF880TK3-1/4")
        assert "%2F" in url  # Slash should be encoded
        assert url.endswith("LF880TK3-1%2F4")

    def test_part_number_with_spaces(self):
        config = {"search_url_template": "https://example.com/search?q={part_number}"}
        url = _build_product_url(config, "HJ 218 EC")
        assert "%20" in url  # Spaces should be encoded


class TestScrapeManufacturerSite:
    """Test the main entry point returns [] for non-approved manufacturers."""

    def test_no_driver_returns_empty(self):
        product = {"mfr_name": "SKF", "mfr_part_number": "21315"}
        assert scrape_manufacturer_site(product, driver=None) == []

    def test_unknown_manufacturer_returns_empty(self):
        product = {"mfr_name": "UNKNOWN CORP", "mfr_part_number": "12345"}
        driver = MagicMock()
        assert scrape_manufacturer_site(product, driver=driver) == []

    def test_non_approved_manufacturer_returns_empty(self):
        """All manufacturers start as non-approved, so should return []."""
        product = {"mfr_name": "SKF", "mfr_part_number": "21315 E/C3"}
        driver = MagicMock()
        result = scrape_manufacturer_site(product, driver=driver)
        assert result == []

    def test_all_registry_entries_not_approved(self):
        """Verify all manufacturers in registry are not approved by default."""
        for key, config in MANUFACTURER_REGISTRY.items():
            assert config["approved"] is False, f"{key} should not be approved by default"

    def test_no_part_number_returns_empty(self):
        """Even if approved, no part number should return []."""
        product = {"mfr_name": "SKF", "mfr_part_number": ""}
        driver = MagicMock()

        # Temporarily approve SKF for this test
        original = MANUFACTURER_REGISTRY["skf"]["approved"]
        MANUFACTURER_REGISTRY["skf"]["approved"] = True
        try:
            result = scrape_manufacturer_site(product, driver=driver)
            assert result == []
        finally:
            MANUFACTURER_REGISTRY["skf"]["approved"] = original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

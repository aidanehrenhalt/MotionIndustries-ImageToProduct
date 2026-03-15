"""
Unit tests for manufacturer_scrapers.py
"""

import pytest
from unittest.mock import MagicMock, patch
from manufacturer_scrapers import (
    _normalize_mfr_name,
    check_robots_txt,
    _build_product_url,
    _extract_images_from_html,
    scrape_manufacturer_site,
    scrape_manufacturer_images,
    MANUFACTURER_REGISTRY,
    CAD_EXTENSIONS,
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
    """Test robots.txt checking with mocked requests.get + RobotFileParser."""

    def setup_method(self):
        # Clear cache between tests
        _robots_cache.clear()

    @patch("manufacturer_scrapers.requests.get")
    @patch("manufacturer_scrapers.RobotFileParser")
    def test_allowed_url(self, mock_rp_class, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "User-agent: *\nAllow: /"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = True
        mock_rp_class.return_value = mock_rp

        assert check_robots_txt("https://www.skf.com/us/products?q=test") is True
        mock_rp.can_fetch.assert_called_once()

    @patch("manufacturer_scrapers.requests.get")
    @patch("manufacturer_scrapers.RobotFileParser")
    def test_disallowed_url(self, mock_rp_class, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "User-agent: *\nDisallow: /"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = False
        mock_rp_class.return_value = mock_rp

        assert check_robots_txt("https://www.skf.com/admin") is False

    @patch("manufacturer_scrapers.requests.get", side_effect=Exception("Connection refused"))
    @patch("manufacturer_scrapers.RobotFileParser")
    def test_fetch_error_returns_false(self, mock_rp_class, mock_get):
        assert check_robots_txt("https://unreachable.example.com/page") is False

    @patch("manufacturer_scrapers.requests.get")
    @patch("manufacturer_scrapers.RobotFileParser")
    def test_caches_per_domain(self, mock_rp_class, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "User-agent: *\nAllow: /"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = True
        mock_rp_class.return_value = mock_rp

        check_robots_txt("https://www.skf.com/page1")
        check_robots_txt("https://www.skf.com/page2")

        # robots.txt fetched once, parser created once for same domain
        assert mock_get.call_count == 1
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

    def test_pending_registry_entries_not_approved(self):
        """Only ami bearings inc and ntn are approved; all others must be False."""
        tier1_approved = {"ami bearings inc", "ntn"}
        for key, config in MANUFACTURER_REGISTRY.items():
            if key not in tier1_approved:
                assert config["approved"] is False, f"{key} should not be approved yet"

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


class TestTier1Registry:
    """Test Tier 1 registry entries have correct structure."""

    def test_ami_registry_approved(self):
        """AMI Bearings entry should be approved with requests renderer."""
        config = MANUFACTURER_REGISTRY["ami bearings inc"]
        assert config["approved"] is True
        assert config["renderer"] == "requests"
        assert len(config["image_selectors"]) > 0

    def test_ntn_registry_approved(self):
        """NTN entry should be approved with requests renderer."""
        config = MANUFACTURER_REGISTRY["ntn"]
        assert config["approved"] is True
        assert config["renderer"] == "requests"

    def test_timken_registry_not_approved(self):
        """Timken is not approved: /keyword/ path is disallowed by robots.txt."""
        config = MANUFACTURER_REGISTRY["timken"]
        assert config["approved"] is False
        assert "cad.timken.com" in config["base_url"]

    def test_timken_selectors_no_cad_extensions(self):
        """Timken selectors must not reference CAD file extensions directly."""
        config = MANUFACTURER_REGISTRY["timken"]
        for selector in config["image_selectors"]:
            for ext in CAD_EXTENSIONS:
                assert ext not in selector.lower(), (
                    f"Timken selector '{selector}' references CAD extension '{ext}'"
                )

    def test_skf_still_not_approved(self):
        """SKF should remain unapproved pending robots.txt + ToS review."""
        assert MANUFACTURER_REGISTRY["skf"]["approved"] is False


class TestAliasesForNewEntries:
    """
    Test _ALIASES correctly routes MFR_NAME column values to registry keys.
    All alias keys must be values from the MFR_NAME column (not ENTERPRISE_NAME),
    since the two CSV columns are independent lists with no row-level relationship.
    """

    def test_timken_fafnir_alias(self):
        assert _normalize_mfr_name("TIMKEN (FAFNIR)") == "timken"

    def test_timken_belts_carlisle_alias(self):
        assert _normalize_mfr_name("TIMKEN BELTS (CARLISLE)") == "timken"

    def test_timken_drives_llc_alias(self):
        assert _normalize_mfr_name("TIMKEN DRIVES LLC") == "timken"

    def test_timken_national_seals_alias(self):
        assert _normalize_mfr_name("TIMKEN NATIONAL SEALS") == "timken"

    def test_ntn_direct_mfr_name(self):
        """'NTN' is the actual MFR_NAME column value — not 'NTN BEARING GROUP'."""
        assert _normalize_mfr_name("NTN") == "ntn"

    def test_ntn_bearing_group_is_enterprise_name_not_aliased(self):
        """'NTN BEARING GROUP' is an ENTERPRISE_NAME, not a MFR_NAME — should not map to 'ntn'."""
        result = _normalize_mfr_name("NTN BEARING GROUP")
        assert result == "ntn bearing group"  # no match, returned as-is (normalized)

    def test_ami_bearings_inc_direct(self):
        assert _normalize_mfr_name("AMI BEARINGS INC") == "ami bearings inc"

    def test_ami_bearings_short_alias(self):
        assert _normalize_mfr_name("AMI BEARINGS") == "ami bearings inc"

    def test_rexnord_inc_mfr_name_alias(self):
        """'REXNORD INC' appears in MFR_NAME column — valid alias."""
        assert _normalize_mfr_name("REXNORD INC") == "rexnord"

    def test_regal_rexnord_is_enterprise_name_not_aliased(self):
        """'REGAL-REXNORD' is an ENTERPRISE_NAME only — should not map to 'rexnord'."""
        result = _normalize_mfr_name("REGAL-REXNORD")
        assert result == "regal-rexnord"  # no match, returned as-is

    def test_skf_sub_brand_cooper_bearings(self):
        """'COOPER BEARINGS (SKF)' appears in MFR_NAME column."""
        assert _normalize_mfr_name("COOPER BEARINGS (SKF)") == "skf"

    def test_skf_sub_brand_mrc(self):
        assert _normalize_mfr_name("MRC (SKF)") == "skf"

    def test_nsk_variant(self):
        assert _normalize_mfr_name("NSK CORP, BEARING DIV") == "nsk"


class TestExtractImagesFromHtml:
    """Test _extract_images_from_html with various HTML inputs."""

    _PRODUCT = {"mfr_name": "AMI BEARINGS INC", "motion_product_id": "test-001"}
    _CONFIG = {
        "image_selectors": ["a[href^='/Asset/']", "img[src^='/Asset/']"],
        "license_note": "Proprietary",
    }

    def test_extracts_asset_link(self):
        html = '<html><body><a href="/Asset/UCT300.jpg">img</a></body></html>'
        images = _extract_images_from_html(
            html, self._CONFIG, self._PRODUCT,
            "https://catalog.amibearings.com",
            "https://catalog.amibearings.com/search?q=UCT300",
        )
        assert len(images) == 1
        assert "UCT300.jpg" in images[0]["image_url"]
        assert images[0]["_source_fn"] == "manufacturer_tier1"

    def test_skips_svg(self):
        html = '<html><body><a href="/Asset/icon.svg">svg</a></body></html>'
        images = _extract_images_from_html(
            html, self._CONFIG, self._PRODUCT,
            "https://catalog.amibearings.com",
            "https://catalog.amibearings.com/search?q=UCT300",
        )
        assert images == []

    def test_skips_cad_extensions(self):
        """Any CAD file extension must be filtered even if selector matches."""
        for ext in [".step", ".stp", ".dxf", ".igs"]:
            html = f'<html><body><a href="/Asset/part{ext}">cad</a></body></html>'
            images = _extract_images_from_html(
                html, self._CONFIG, self._PRODUCT,
                "https://catalog.amibearings.com",
                "https://catalog.amibearings.com/search?q=test",
            )
            assert images == [], f"CAD extension '{ext}' was not filtered"

    def test_skips_data_uri(self):
        html = '<html><body><img src="data:image/png;base64,abc"></body></html>'
        config = {"image_selectors": ["img"], "license_note": "Proprietary"}
        images = _extract_images_from_html(
            html, config, self._PRODUCT,
            "https://example.com", "https://example.com/page",
        )
        assert images == []

    def test_deduplicates_urls(self):
        html = (
            '<html><body>'
            '<a href="/Asset/UCT300.jpg">1</a>'
            '<a href="/Asset/UCT300.jpg">2</a>'
            '</body></html>'
        )
        images = _extract_images_from_html(
            html, self._CONFIG, self._PRODUCT,
            "https://catalog.amibearings.com",
            "https://catalog.amibearings.com/search?q=UCT300",
        )
        assert len(images) == 1

    def test_respects_max_images_per_mfr(self):
        links = "".join(f'<a href="/Asset/img{i}.jpg">x</a>' for i in range(10))
        html = f"<html><body>{links}</body></html>"
        images = _extract_images_from_html(
            html, self._CONFIG, self._PRODUCT,
            "https://catalog.amibearings.com",
            "https://catalog.amibearings.com/search?q=test",
        )
        assert len(images) == 3  # MAX_IMAGES_PER_MFR


class TestScrapeManufacturerImages:
    """Test the new scrape_manufacturer_images() public entry point."""

    def setup_method(self):
        _robots_cache.clear()

    def test_not_approved_returns_empty(self):
        """SKF is in the registry but not approved — must return []."""
        product = {"mfr_name": "SKF", "mfr_part_number": "21315"}
        assert scrape_manufacturer_images(product) == []

    def test_unknown_manufacturer_tier2_disabled_returns_empty(self):
        """Unknown manufacturer with enable_tier2=False must return []."""
        product = {"mfr_name": "TOTALLY UNKNOWN CORP", "mfr_part_number": "X999"}
        assert scrape_manufacturer_images(product, enable_tier2=False) == []

    @patch("manufacturer_scrapers.requests.get")
    @patch("manufacturer_scrapers.check_robots_txt", return_value=True)
    def test_tier1_ami_calls_requests(self, mock_robots, mock_get):
        """Approved Tier 1 entry (AMI) should use requests, not Selenium."""
        mock_resp = MagicMock()
        mock_resp.text = '<html><body><a href="/Asset/UCT300.jpg">img</a></body></html>'
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        product = {"mfr_name": "AMI BEARINGS INC", "mfr_part_number": "UCT300", "motion_product_id": "t1"}
        images = scrape_manufacturer_images(product)
        assert len(images) == 1
        assert mock_get.called
        assert images[0]["_source_fn"] == "manufacturer_tier1"

    @patch("manufacturer_scrapers._scrape_tier2_generic")
    def test_tier2_called_for_unknown_manufacturer(self, mock_tier2):
        """Unknown manufacturer with enable_tier2=True should invoke _scrape_tier2_generic."""
        mock_tier2.return_value = []
        product = {"mfr_name": "TOTALLY UNKNOWN CORP", "mfr_part_number": "X999"}
        scrape_manufacturer_images(product, enable_tier2=True)
        mock_tier2.assert_called_once_with(product)

    def test_tier2_playwright_import_error_returns_empty(self):
        """If playwright is not installed, Tier 2 must return [] gracefully."""
        product = {"mfr_name": "UNKNOWN CORP", "mfr_part_number": "X123"}
        with patch.dict("sys.modules", {"playwright": None, "playwright.sync_api": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'playwright'")):
                result = scrape_manufacturer_images(product, enable_tier2=True)
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

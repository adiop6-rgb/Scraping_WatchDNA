#!/usr/bin/env python3
"""AI-driven watch scraper that adapts to arbitrary brand catalogs.

This tool uses Google's Gemini model to interpret arbitrary watch product
pages and emit Shopify-compatible CSV rows without maintaining per-brand
CSS selector maps. It fetches catalog/product HTML, provides the rendered
content plus extracted cues to the model, and normalises the response to the
Shopify export schema captured from products_export.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from contextlib import contextmanager

import requests
from bs4 import BeautifulSoup

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover - dependency notice
    raise SystemExit(
        "Missing google-generativeai. Install deps via 'pip install google-generativeai beautifulsoup4 requests'."
    ) from exc

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright
except ImportError:  # pragma: no cover - optional dependency
    PlaywrightTimeoutError = Exception  # type: ignore[assignment]
    sync_playwright = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Local environment loading
# ---------------------------------------------------------------------------

# Point to the folder holding this script so we can find helper files beside it.
SCRIPT_DIR = Path(__file__).resolve().parent
# This optional file lets the scraper load private keys without hard-coding them.
LOCAL_ENV_FILE = SCRIPT_DIR / ".env"


def load_local_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from the scraper-local .env file."""

    # Skip quietly when there is no local secrets file.
    if not env_path.exists():
        return

    try:
        lines = env_path.read_text().splitlines()
    except OSError:
        # Reading may fail on locked files; let the script continue anyway.
        return

    for raw_line in lines:
        # Clean the line so only useful KEY=VALUE entries remain.
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or key in os.environ:
            continue
        os.environ[key] = value


# Import user-provided keys before touching any network services.
load_local_env_file(LOCAL_ENV_FILE)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Choose the model and API key sources so the scraper can run anywhere.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
ENV_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GENAI_API_KEY")
    or os.getenv("API_KEY")
)
# Set this if you prefer baking the key directly into the script.
HARDCODED_GOOGLE_API_KEY = ""

GOOGLE_API_KEY = HARDCODED_GOOGLE_API_KEY or ENV_API_KEY

# These baseline values match Shopify's defaults when the site leaves blanks.
SHOPIFY_DEFAULTS: Dict[str, str] = {
    "Published": "true",
    "Status": "active",
    "Gift Card": "false",
    "Variant Inventory Tracker": "shopify",
    "Variant Inventory Policy": "continue",
    "Variant Fulfillment Service": "manual",
    "Variant Inventory Qty": "1",
    "Product Category": "Apparel & Accessories > Jewelry > Watches",
    "Google Shopping / Google Product Category": "201",
}

REQUIRED_PRODUCT_CATEGORY = "Apparel & Accessories > Jewelry > Watches"
REQUIRED_PRODUCT_TYPE = "Wrist Watch"

# Keep the header order identical to Shopify's export for smooth imports.
SHOPIFY_HEADERS: List[str] = [
    "Handle",
    "Title",
    "Body (HTML)",
    "Vendor",
    "Product Category",
    "Type",
    "Tags",
    "Published",
    "Option1 Name",
    "Option1 Value",
    "Option1 Linked To",
    "Option2 Name",
    "Option2 Value",
    "Option2 Linked To",
    "Option3 Name",
    "Option3 Value",
    "Option3 Linked To",
    "Variant SKU",
    "Variant Grams",
    "Variant Inventory Tracker",
    "Variant Inventory Qty",
    "Variant Inventory Policy",
    "Variant Fulfillment Service",
    "Variant Price",
    "Variant Compare At Price",
    "Variant Requires Shipping",
    "Variant Taxable",
    "Unit Price Total Measure",
    "Unit Price Total Measure Unit",
    "Unit Price Base Measure",
    "Unit Price Base Measure Unit",
    "Variant Barcode",
    "Image Src",
    "Image Position",
    "Image Alt Text",
    "Gift Card",
    "SEO Title",
    "SEO Description",
    "Strap Color (product.metafields.custom.strap_color)",
    "Case Material (product.metafields.custom.case_material)",
    "Collection (product.metafields.custom.collection)",
    "Edition Type (product.metafields.custom.edition_type)",
    "Movement (product.metafields.custom.movement)",
    "Strap (product.metafields.custom.strap)",
    "Style (product.metafields.custom.style)",
    "Watch display (product.metafields.custom.watch_display)",
    "Watch features (product.metafields.custom.watch_features)",
    "Water Resistance (product.metafields.custom.water_resistance)",
    "Color (product.metafields.shopify.color-pattern)",
    "Dial color (product.metafields.shopify.dial-color)",
    "Complementary products (product.metafields.shopify--discovery--product_recommendation.complementary_products)",
    "Related products (product.metafields.shopify--discovery--product_recommendation.related_products)",
    "Related products settings (product.metafields.shopify--discovery--product_recommendation.related_products_display)",
    "Search product boosts (product.metafields.shopify--discovery--product_search_boost.queries)",
    "Variant Image",
    "Variant Weight Unit",
    "Variant Tax Code",
    "Cost per item",
    "Status",
]

FIELD_POLICY_LABELS: Dict[str, str] = {
    "fill": "FILL",
    "opportunistic": "OPPORTUNISTIC",
    "static": "STATIC",
}

# Plain-language rules tell the model how confident it should be before filling a field.
SHOPIFY_FIELD_GUIDANCE: Dict[str, Dict[str, str]] = {
    "Handle": {
        "policy": "fill",
        "description": "Canonical handle/slug. Use the on-page handle or derive from the URL (lowercase, hyphenated).",
    },
    "Title": {
        "policy": "fill",
        "description": "Marketing name exactly as shown on the product page.",
    },
    "Body (HTML)": {
        "policy": "fill",
        "description": "Full product description and specs preserved in HTML (paragraphs, lists, tables).",
    },
    "Vendor": {
        "policy": "fill",
        "description": "Brand or manufacturer name displayed on the site.",
    },
    "Product Category": {
        "policy": "static",
        "description": "Hard-code to Apparel & Accessories > Jewelry > Watches.",
    },
    "Type": {
        "policy": "fill",
        "description": "Shopify product type (e.g., Chronograph, Dive Watch). Use the brand's wording when possible.",
    },
    "Tags": {
        "policy": "opportunistic",
        "description": "Comma-separated descriptive keywords pulled from visible tag lists or obvious descriptors.",
    },
    "Published": {
        "policy": "static",
        "description": "Leave blank. Importer keeps the default 'true' published state.",
    },
    "Option1 Name": {
        "policy": "static",
        "description": "Reserved for the CONNECT button link. Use 'BRAND URL' when a safe https:// link is available.",
    },
    "Option1 Value": {
        "policy": "static",
        "description": "HTML anchor tag (<a href=\"https://...\">CONNECT</a>) when safe; otherwise fallback to Default Title.",
    },
    "Option1 Linked To": {
        "policy": "static",
        "description": "Leave blank. Shopify manages option linkage internally.",
    },
    "Option2 Name": {
        "policy": "static",
        "description": "Leave blank. Option2 is not used in the no-variants model.",
    },
    "Option2 Value": {
        "policy": "static",
        "description": "Leave blank. Option2 is not used in the no-variants model.",
    },
    "Option2 Linked To": {
        "policy": "static",
        "description": "Leave blank; only admin tooling configures this mapping.",
    },
    "Option3 Name": {
        "policy": "static",
        "description": "Leave blank. Option3 is not used in the no-variants model.",
    },
    "Option3 Value": {
        "policy": "static",
        "description": "Leave blank. Option3 is not used in the no-variants model.",
    },
    "Option3 Linked To": {
        "policy": "static",
        "description": "Leave blank; Shopify links options internally.",
    },
    "Variant SKU": {
        "policy": "opportunistic",
        "description": "Reference/stock number exposed on the page. Leave blank if no explicit SKU/reference.",
    },
    "Variant Grams": {
        "policy": "opportunistic",
        "description": "Mass in grams as an integer (e.g., 227) when provided. Otherwise blank.",
    },
    "Variant Inventory Tracker": {
        "policy": "static",
        "description": "Leave blank; importer enforces the default 'shopify' tracker.",
    },
    "Variant Inventory Qty": {
        "policy": "static",
        "description": "Leave blank; importer sets a static quantity (1).",
    },
    "Variant Inventory Policy": {
        "policy": "static",
        "description": "Leave blank; importer supplies the default 'continue' policy.",
    },
    "Variant Fulfillment Service": {
        "policy": "static",
        "description": "Leave blank; importer applies the default 'manual' service.",
    },
    "Variant Price": {
        "policy": "fill",
        "description": "Current sell price as a plain number (no currency symbols).",
    },
    "Variant Compare At Price": {
        "policy": "fill",
        "description": "List/original price when the site shows an MSRP or crossed-out price.",
    },
    "Variant Requires Shipping": {
        "policy": "static",
        "description": "Leave blank; importer defaults watches to require shipping.",
    },
    "Variant Taxable": {
        "policy": "static",
        "description": "Leave blank; importer defaults to taxable goods.",
    },
    "Unit Price Total Measure": {
        "policy": "static",
        "description": "Not exposed publicly. Leave blank.",
    },
    "Unit Price Total Measure Unit": {
        "policy": "static",
        "description": "Not exposed publicly. Leave blank.",
    },
    "Unit Price Base Measure": {
        "policy": "static",
        "description": "Not exposed publicly. Leave blank.",
    },
    "Unit Price Base Measure Unit": {
        "policy": "static",
        "description": "Not exposed publicly. Leave blank.",
    },
    "Variant Barcode": {
        "policy": "opportunistic",
        "description": "UPC/EAN barcodes when the storefront reveals them.",
    },
    "Image Src": {
        "policy": "fill",
        "description": "Absolute URL of the hero image or first gallery image.",
    },
    "Image Position": {
        "policy": "fill",
        "description": "Set to '1' for the hero image captured above.",
    },
    "Image Alt Text": {
        "policy": "fill",
        "description": "Short descriptive alt text referencing the watch and key traits.",
    },
    "Gift Card": {
        "policy": "static",
        "description": "Leave blank; importer enforces 'false'.",
    },
    "SEO Title": {
        "policy": "fill",
        "description": "SEO-safe title (often mirrors the product title).",
    },
    "SEO Description": {
        "policy": "fill",
        "description": "320-char meta description distilled from the page content.",
    },
    "Google Shopping / Google Product Category": {
        "policy": "static",
        "description": "Leave blank; importer applies the watch category code.",
    },
    "Google Shopping / Gender": {
        "policy": "opportunistic",
        "description": "Target gender when explicitly stated (e.g., Men's, Women's, Unisex).",
    },
    "Google Shopping / Age Group": {
        "policy": "opportunistic",
        "description": "Age group tag (Adult, Teen, Kids) when visible.",
    },
    "Google Shopping / MPN": {
        "policy": "opportunistic",
        "description": "Manufacturer part/reference number pulled from specs.",
    },
    "Google Shopping / Condition": {
        "policy": "opportunistic",
        "description": "Condition label (New, Pre-Owned) when explicitly shown.",
    },
    "Google Shopping / Custom Product": {
        "policy": "static",
        "description": "Marketing flag managed in Shopify admin. Leave blank.",
    },
    "Google Shopping / Custom Label 0": {
        "policy": "static",
        "description": "Leave blank; merchandising team configures labels.",
    },
    "Google Shopping / Custom Label 1": {
        "policy": "static",
        "description": "Leave blank; merchandising team configures labels.",
    },
    "Google Shopping / Custom Label 2": {
        "policy": "static",
        "description": "Leave blank; merchandising team configures labels.",
    },
    "Google Shopping / Custom Label 3": {
        "policy": "static",
        "description": "Leave blank; merchandising team configures labels.",
    },
    "Google Shopping / Custom Label 4": {
        "policy": "static",
        "description": "Leave blank; merchandising team configures labels.",
    },
    "Accessories type (product.metafields.custom.accessory_type)": {
        "policy": "opportunistic",
        "description": "Accessory classification (e.g., Watch Roll, Tool) when the product is bundled with extras.",
    },
    "Case Material (product.metafields.custom.case_material)": {
        "policy": "fill",
        "description": "Case construction material (e.g., Stainless steel, Ceramic).",
    },
    "Collection (product.metafields.custom.collection)": {
        "policy": "opportunistic",
        "description": "Collection/family name (e.g., Carrera, Seamaster) when stated.",
    },
    "Condition (product.metafields.custom.condition)": {
        "policy": "opportunistic",
        "description": "Condition wording when the storefront differentiates new vs pre-owned.",
    },
    "Edition Type (product.metafields.custom.edition_type)": {
        "policy": "opportunistic",
        "description": "Edition notes (Limited, Special, Anniversary) from the copy.",
    },
    "Gender (product.metafields.custom.gender)": {
        "policy": "static",
        "description": "Leave blank; use the Shopify target-gender metafield instead.",
    },
    "Movement (product.metafields.custom.movement)": {
        "policy": "fill",
        "description": "Movement category only (Automatic, Quartz, Manual, Hybrid).",
    },
    "Strap (product.metafields.custom.strap)": {
        "policy": "fill",
        "description": "Strap or bracelet material/type (e.g., Rubber strap, Steel bracelet).",
    },
    "Style (product.metafields.custom.style)": {
        "policy": "opportunistic",
        "description": "Style descriptors (dress, sport, pilot) lifted from the copy.",
    },
    "Water Resistance (product.metafields.custom.water_resistance)": {
        "policy": "fill",
        "description": "Water resistance rating (e.g., 100 m / 10 bar) from the specs.",
    },
    "LDT: Compare Attribute Set Code (product.metafields.ldt.compare_attribute_set)": {
        "policy": "static",
        "description": "Internal comparison metadata. Leave blank.",
    },
    "LDT: Compare Products (product.metafields.ldt.compare_products)": {
        "policy": "static",
        "description": "Internal comparison metadata. Leave blank.",
    },
    "Product rating count (product.metafields.reviews.rating_count)": {
        "policy": "opportunistic",
        "description": "Visible review count when the page exposes ratings.",
    },
    "Target gender (product.metafields.shopify.target-gender)": {
        "policy": "fill",
        "description": "Strict enum: Men, Women, or Unisex only.",
    },
    "Age group (product.metafields.shopify.age-group)": {
        "policy": "fill",
        "description": "Strict enum: adult or child only.",
    },
    "Strap Color (product.metafields.custom.strap_color)": {
        "policy": "fill",
        "description": "Single strap/bracelet color name.",
    },
    "Color (product.metafields.shopify.color-pattern)": {
        "policy": "static",
        "description": "Leave blank; dial and band colors are captured in dedicated custom metafields.",
    },
    "Dial color (product.metafields.shopify.dial-color)": {
        "policy": "static",
        "description": "Leave blank; not populated for this store.",
    },
    "Watch display (product.metafields.custom.watch_display)": {
        "policy": "fill",
        "description": "Display type (analog, digital, skeleton, etc.).",
    },
    "Watch features (product.metafields.custom.watch_features)": {
        "policy": "fill",
        "description": "Semicolon-separated short tokens (e.g., chronograph; gmt; 10atm).",
    },
    "Complementary products (product.metafields.shopify--discovery--product_recommendation.complementary_products)": {
        "policy": "opportunistic",
        "description": "Complementary SKUs only when the page explicitly lists them.",
    },
    "Related products (product.metafields.shopify--discovery--product_recommendation.related_products)": {
        "policy": "opportunistic",
        "description": "Related SKUs when the storefront surfaces specific product IDs.",
    },
    "Related products settings (product.metafields.shopify--discovery--product_recommendation.related_products_display)": {
        "policy": "static",
        "description": "Admin-only discovery settings. Leave blank.",
    },
    "Search product boosts (product.metafields.shopify--discovery--product_search_boost.queries)": {
        "policy": "static",
        "description": "Admin search boost controls. Leave blank.",
    },
    "Variant Image": {
        "policy": "fill",
        "description": "Image URL tied to the variant (reuse the hero image for single variants).",
    },
    "Variant Weight Unit": {
        "policy": "static",
        "description": "Leave blank; Variant Grams already stores gram values.",
    },
    "Variant Tax Code": {
        "policy": "static",
        "description": "Tax code managed in Shopify admin. Leave blank.",
    },
    "Cost per item": {
        "policy": "static",
        "description": "COGS not published on storefronts. Leave blank.",
    },
    "Status": {
        "policy": "static",
        "description": "Leave blank; importer enforces the default 'active' status.",
    },
}

_MISSING_GUIDANCE = [name for name in SHOPIFY_HEADERS if name not in SHOPIFY_FIELD_GUIDANCE]
if _MISSING_GUIDANCE:
    raise RuntimeError(f"Missing field guidance for: {', '.join(_MISSING_GUIDANCE)}")
# These raw keys describe the clean facts we expect back from the model.
RAW_FIELD_SPECS: Dict[str, str] = {
    "is_product_page": (
        "Boolean flag. Set to true only when this page describes a single watch product "
        "(one SKU/variant family). Set to false when the page is a brand homepage, "
        "category/collection, multi-product listing, blog, or any non-product content."
    ),
    "variants": (
        "Optional list of variant objects when the product has multiple variants (e.g., strap or dial colours). "
        "Each variant object may include: 'sku', 'price', 'compare_at_price', 'grams', 'requires_shipping', "
        "'taxable', 'barcode', 'options' (list of {name, value}), 'images' (list of image URLs for that variant), "
        "and 'image_alt_text'. Use an empty list when the site does not expose distinct variants."
    ),
    "handle": "Canonical product handle/slug if the page exposes one; otherwise empty.",
    "title": "Primary product name exactly as written.",
    "body_html": "Rich text/HTML body copy; preserve markup like <p> when present.",
    "plain_description": "Fallback plain-text summary when no HTML exists.",
    "seo_description": "Optional SEO-friendly summary (max 320 chars) if provided.",
    "vendor": "Brand or manufacturer.",
    "product_type": "Type/category labels stated by the site.",
    "product_category": "Site's merchandising category hierarchy if shown.",
    "collection": "Collection/family name if mentioned.",
    "reference_number": (
        "Reference/model number or SKU. Prefer alphanumeric codes (e.g. 'NZSSWH', 'H70455553') found "
        "at the start of the description or in specs. Ignore small integers (like '10', '20') or dimensions."
    ),
    "barcode": "UPC/EAN if shown.",
    "price": "Current sell price numeric (no currency symbols).",
    "compare_at_price": "Original MSRP if shown; else empty.",
    "currency": "Currency code (e.g., CHF, USD) if visible.",
    "grams": "Mass in grams as a pure number.",
    "requires_shipping": "true/false if explicitly stated; else empty.",
    "taxable": "true/false if mentioned; else empty.",
    "tags": "List of descriptive keywords or hashtags.",
    "options": (
        "List of variant option objects with keys 'name' and 'value'. "
        "One object per option (e.g., size, strap). Use the same wording as the site."
    ),
    "images": "List of absolute URLs for the best product images (max 8).",
    "image_alt_text": "Alt text or caption describing the hero image.",
    "case_material": "Case material description.",
    "case_size": "Case diameter or size text.",
    "case_length": "Case length or lug-to-lug measurement text with units.",
    "watch_case_diameter": "Case diameter/width measurement with units.",
    "case_thickness": "Case thickness/depth measurement with units.",
    "movement": (
        "Movement TYPE/CATEGORY only. Use one of: Automatic, Manual, Quartz, Hybrid. "
        "Do NOT include calibre names/codes or long descriptions here."
    ),
    "caliber_type": (
        "Movement calibre name/code only (e.g., 'Breitling Calibre B01', 'NH35A', 'SW200-1', 'ETA 2824-2'). "
        "Do NOT include generic movement-type words like Automatic/Quartz/Manual."
    ),
    "strap_type": "Bracelet/strap type or material.",
    "strap_color": "Strap color.",
    "dial_color": "Dial color.",
    "water_resistance": "Water resistance rating.",
    "water_resistance_m": "Water resistance in meters (e.g., '100 m').",
    "other_features": "Array of notable features/highlights (one string per bullet).",
    "complications": "List of complications displayed in the specs.",
    "gender": "Intended gender if explicitly stated.",
    "age_group": "Intended age group if explicitly stated (adult or child).",
    "style": "Style descriptors (sport, dress, etc.).",
    "condition": "Condition wording if listed (e.g., New).",
    "edition_type": "Edition info (limited, special, etc.).",
    "watch_display": "Watch display type (analog, digital, etc.).",
    "feature_pairs": (
        "List of watch-specification key/value pairs. Each item must be an object with keys "
        "'label' and 'value'. Only include technical watch features/specs (case, dial, movement, "
        "crystal, bezel, water resistance, strap/bracelet, dimensions, complications/functions). "
        "Exclude marketing copy, lifestyle text, shipping/delivery, returns/refunds, warranty, "
        "payment/financing, reviews, and store policies. Use an empty list when no explicit "
        "specification pairs are present."
    ),
}


@dataclass(frozen=True)
class CanonicalWatchFeature:
    key: str
    label: str
    namespace: str = "custom"


# Canonical watch feature schema used for consistent metafield naming.
# Notes:
# - Keys must be Shopify metafield-safe: lowercase snake_case.
# - Labels should be stable across all brand exports.
CANONICAL_WATCH_FEATURES: Tuple[CanonicalWatchFeature, ...] = (
    # Case
    CanonicalWatchFeature("case_material", "Case Material"),
    CanonicalWatchFeature("case_shape", "Case Shape"),
    CanonicalWatchFeature("case_color", "Case Color"),
    CanonicalWatchFeature("case_finish", "Case Finish"),
    CanonicalWatchFeature("case_coating", "Case Coating"),
    CanonicalWatchFeature("case_diameter", "Case Diameter"),
    CanonicalWatchFeature("case_thickness", "Case Thickness"),
    CanonicalWatchFeature("lug_to_lug", "Lug-to-Lug"),
    CanonicalWatchFeature("lug_width", "Lug Width"),
    CanonicalWatchFeature("caseback_type", "Caseback Type"),
    CanonicalWatchFeature("caseback_material", "Caseback Material"),
    CanonicalWatchFeature("crown_type", "Crown Type"),
    CanonicalWatchFeature("crown_material", "Crown Material"),
    CanonicalWatchFeature("bezel_material", "Bezel Material"),
    CanonicalWatchFeature("bezel_type", "Bezel Type"),
    CanonicalWatchFeature("bezel_color", "Bezel Color"),
    CanonicalWatchFeature("bezel_function", "Bezel Function"),
    CanonicalWatchFeature("bezel_insert", "Bezel Insert"),
    CanonicalWatchFeature("crystal_material", "Crystal Material"),
    CanonicalWatchFeature("crystal_shape", "Crystal Shape"),
    CanonicalWatchFeature("crystal_coating", "Crystal Coating"),
    CanonicalWatchFeature("water_resistance", "Water Resistance"),
    CanonicalWatchFeature("screw_down_crown", "Screw-down Crown"),
    CanonicalWatchFeature("screw_down_pushers", "Screw-down Pushers"),
    CanonicalWatchFeature("helium_escape_valve", "Helium Escape Valve"),
    CanonicalWatchFeature("anti_magnetic", "Antimagnetic"),
    CanonicalWatchFeature("shock_resistance", "Shock Resistance"),
    # Dial
    CanonicalWatchFeature("dial_color", "Dial Color"),
    CanonicalWatchFeature("dial_material", "Dial Material"),
    CanonicalWatchFeature("dial_finish", "Dial Finish"),
    CanonicalWatchFeature("dial_pattern", "Dial Pattern"),
    CanonicalWatchFeature("dial_markers", "Dial Markers"),
    CanonicalWatchFeature("hands", "Hands"),
    CanonicalWatchFeature("hands_color", "Hands Color"),
    CanonicalWatchFeature("lume", "Lume"),
    CanonicalWatchFeature("lume_color", "Lume Color"),
    CanonicalWatchFeature("dial_indices", "Dial Indices"),
    CanonicalWatchFeature("dial_type", "Dial Type"),
    # Movement
    CanonicalWatchFeature("movement", "Movement"),
    CanonicalWatchFeature("caliber", "Caliber"),
    CanonicalWatchFeature("movement_manufacturer", "Movement Manufacturer"),
    CanonicalWatchFeature("winding", "Winding"),
    CanonicalWatchFeature("hacking_seconds", "Hacking Seconds"),
    CanonicalWatchFeature("hand_winding", "Hand Winding"),
    CanonicalWatchFeature("chronometer", "Chronometer"),
    CanonicalWatchFeature("certification", "Certification"),
    CanonicalWatchFeature("power_reserve", "Power Reserve"),
    CanonicalWatchFeature("frequency", "Frequency"),
    CanonicalWatchFeature("jewels", "Jewels"),
    CanonicalWatchFeature("accuracy", "Accuracy"),
    CanonicalWatchFeature("complications", "Complications"),
    CanonicalWatchFeature("functions", "Functions"),
    # Strap / bracelet
    CanonicalWatchFeature("strap", "Strap"),
    CanonicalWatchFeature("strap_material", "Strap Material"),
    CanonicalWatchFeature("strap_color", "Strap Color"),
    CanonicalWatchFeature("strap_width", "Strap Width"),
    CanonicalWatchFeature("bracelet_type", "Bracelet Type"),
    CanonicalWatchFeature("clasp_type", "Clasp Type"),
    CanonicalWatchFeature("clasp_material", "Clasp Material"),
    CanonicalWatchFeature("buckle_type", "Buckle Type"),
    CanonicalWatchFeature("quick_release", "Quick Release"),
    # Misc
    CanonicalWatchFeature("watch_display", "Watch display"),
    CanonicalWatchFeature("weight", "Weight"),
    CanonicalWatchFeature("edition_type", "Edition Type"),
    CanonicalWatchFeature("edition_number", "Edition Number"),
    CanonicalWatchFeature("country_of_origin", "Country of Origin"),
)

CANONICAL_WATCH_FEATURE_BY_KEY: Dict[str, CanonicalWatchFeature] = {
    feature.key: feature for feature in CANONICAL_WATCH_FEATURES
}


_NON_FEATURE_LABEL_PATTERNS = [
    re.compile(r"\bshipping\b", re.IGNORECASE),
    re.compile(r"\bdelivery\b", re.IGNORECASE),
    re.compile(r"\breturns?\b", re.IGNORECASE),
    re.compile(r"\brefunds?\b", re.IGNORECASE),
    re.compile(r"\bwarranty\b", re.IGNORECASE),
    re.compile(r"\bpayment\b", re.IGNORECASE),
    re.compile(r"\bfinancing\b", re.IGNORECASE),
    re.compile(r"\bcontact\b", re.IGNORECASE),
    re.compile(r"\bnewsletter\b", re.IGNORECASE),
    re.compile(r"\breviews?\b", re.IGNORECASE),
]


def _normalise_feature_label(label: str) -> str:
    label = (label or "").strip().lower()
    label = re.sub(r"[\s:/_-]+", " ", label)
    label = re.sub(r"[^\w\s]", "", label)
    return re.sub(r"\s+", " ", label).strip()


_FEATURE_LABEL_UNIT_SUFFIXES = {
    "mm",
    "cm",
    "m",
    "meter",
    "meters",
    "atm",
    "bar",
    "g",
    "gram",
    "grams",
    "kg",
    "in",
    "inch",
    "inches",
}


def _strip_feature_label_unit_suffix(label: str) -> str:
    tokens = (label or "").split()
    while tokens and tokens[-1] in _FEATURE_LABEL_UNIT_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


_CANONICAL_LABEL_ALIASES: Dict[str, str] = {
    # Case sizing
    "case diameter": "case_diameter",
    "diameter": "case_diameter",
    "case size": "case_diameter",
    "case thickness": "case_thickness",
    "thickness": "case_thickness",
    "height": "case_thickness",
    "depth": "case_thickness",
    "lug to lug": "lug_to_lug",
    "lugtolug": "lug_to_lug",
    "lug width": "lug_width",
    "between lugs": "lug_width",
    # Materials
    "case material": "case_material",
    "bezel material": "bezel_material",
    "caseback material": "caseback_material",
    "case back material": "caseback_material",
    "crystal": "crystal_material",
    "crystal material": "crystal_material",
    "crystal shape": "crystal_shape",
    "crystal coating": "crystal_coating",
    "glass": "crystal_material",
    "crown material": "crown_material",
    # Bezel
    "bezel function": "bezel_function",
    "bezel insert": "bezel_insert",
    # Water resistance
    "water resistance": "water_resistance",
    "water resistant": "water_resistance",
    "wr": "water_resistance",
    "screw down crown": "screw_down_crown",
    "screw-down crown": "screw_down_crown",
    "screw down pushers": "screw_down_pushers",
    "screw-down pushers": "screw_down_pushers",
    "helium escape valve": "helium_escape_valve",
    "hev": "helium_escape_valve",
    "antimagnetic": "anti_magnetic",
    "anti magnetic": "anti_magnetic",
    "shock resistance": "shock_resistance",
    "shock resistant": "shock_resistance",
    # Dial
    "dial color": "dial_color",
    "dial colour": "dial_color",
    "dial pattern": "dial_pattern",
    "dial indices": "dial_indices",
    "indices": "dial_indices",
    "indexes": "dial_indices",
    "index": "dial_indices",
    "dial type": "dial_type",
    "hands color": "hands_color",
    "hands colour": "hands_color",
    "lume color": "lume_color",
    "lume colour": "lume_color",
    "hour markers": "dial_markers",
    # Movement
    "movement type": "movement",
    "movement": "movement",
    "caliber": "caliber",
    "calibre": "caliber",
    "movement manufacturer": "movement_manufacturer",
    "power reserve": "power_reserve",
    "frequency": "frequency",
    "vph": "frequency",
    "jewels": "jewels",
    "accuracy": "accuracy",
    "winding": "winding",
    "winding system": "winding",
    "hacking seconds": "hacking_seconds",
    "stop seconds": "hacking_seconds",
    "hand winding": "hand_winding",
    "manual winding": "hand_winding",
    "chronometer": "chronometer",
    "certification": "certification",
    "certifications": "certification",
    "complications": "complications",
    "functions": "functions",
    # Strap
    "strap": "strap",
    "bracelet": "strap",
    "band": "strap",
    "strap material": "strap_material",
    "bracelet material": "strap_material",
    "strap color": "strap_color",
    "strap colour": "strap_color",
    "band color": "strap_color",
    "band colour": "strap_color",
    "bracelet color": "strap_color",
    "clasp": "clasp_type",
    "clasp type": "clasp_type",
    "clasp material": "clasp_material",
    "bracelet type": "bracelet_type",
    "buckle": "buckle_type",
    "buckle type": "buckle_type",
    "quick release": "quick_release",
    "quick-release": "quick_release",
    # Misc
    "display": "watch_display",
    "watch display": "watch_display",
    "weight": "weight",
    "edition": "edition_type",
    "edition type": "edition_type",
    "limited edition": "edition_type",
    "edition number": "edition_number",
    "limited edition number": "edition_number",
    "edition no": "edition_number",
    "edition no.": "edition_number",
    "country of origin": "country_of_origin",
    "made in": "country_of_origin",
}

_CANONICAL_LABEL_REGEX_RULES: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bcase\b.*\b(material|steel|titanium|ceramic|gold|bronze)\b", re.IGNORECASE), "case_material"),
    (re.compile(r"\b(bezel)\b.*\bmaterial\b", re.IGNORECASE), "bezel_material"),
    (re.compile(r"\bbezel\b.*\b(function|scale|tachymeter|countdown|compass|gmt)\b", re.IGNORECASE), "bezel_function"),
    (re.compile(r"\b(diameter|dia)\b.*\bmm\b", re.IGNORECASE), "case_diameter"),
    (re.compile(r"\b(lug)\b.*\bto\b.*\b(lug)\b", re.IGNORECASE), "lug_to_lug"),
    (re.compile(r"\bwater\b.*\bresist", re.IGNORECASE), "water_resistance"),
    (re.compile(r"\b(power)\b.*\breserve\b", re.IGNORECASE), "power_reserve"),
    (re.compile(r"\bstrap\b.*\b(width)\b", re.IGNORECASE), "strap_width"),
    (re.compile(r"\bquick\b.*\brelease\b", re.IGNORECASE), "quick_release"),
    (re.compile(r"\bbetween\b.*\blugs\b", re.IGNORECASE), "lug_width"),
    (re.compile(r"\bcase\s*back\b.*\bmaterial\b|\bcaseback\b.*\bmaterial\b", re.IGNORECASE), "caseback_material"),
    (re.compile(r"\bcaseback\b", re.IGNORECASE), "caseback_type"),
    (re.compile(r"\bcrown\b.*\bmaterial\b", re.IGNORECASE), "crown_material"),
    (re.compile(r"\bcrown\b", re.IGNORECASE), "crown_type"),
    (re.compile(r"\bcrystal\b|\bglass\b", re.IGNORECASE), "crystal_material"),
    (re.compile(r"\bdial\b.*\b(colou?r)\b", re.IGNORECASE), "dial_color"),
    (re.compile(r"\bdial\b.*\bpattern\b", re.IGNORECASE), "dial_pattern"),
    (re.compile(r"\bdial\b.*\b(indices|indexes|index|numerals)\b", re.IGNORECASE), "dial_indices"),
    (re.compile(r"\bdial\b.*\b(markers|hour markers?)\b", re.IGNORECASE), "dial_markers"),
    (re.compile(r"\bhands?\b.*\b(colou?r)\b", re.IGNORECASE), "hands_color"),
    (re.compile(r"\blume\b.*\b(colou?r)\b", re.IGNORECASE), "lume_color"),
    (re.compile(r"\banti\s*magnetic\b|\bantimagnetic\b", re.IGNORECASE), "anti_magnetic"),
    (re.compile(r"\bshock\b.*\bresist", re.IGNORECASE), "shock_resistance"),
    (re.compile(r"\bhelium\b.*\bescape\b", re.IGNORECASE), "helium_escape_valve"),
    (re.compile(r"\bscrew\b.*\bdown\b.*\bcrown\b", re.IGNORECASE), "screw_down_crown"),
    (re.compile(r"\bscrew\b.*\bdown\b.*\bpushers?\b", re.IGNORECASE), "screw_down_pushers"),
    (re.compile(r"\bmovement\b.*\b(manufacturer|maker)\b", re.IGNORECASE), "movement_manufacturer"),
    (re.compile(r"\bcountry\b.*\borigin\b|\bmade\s+in\b", re.IGNORECASE), "country_of_origin"),
    (re.compile(r"\bmovement\b.*\b(automatic|manual|quartz|solar)\b", re.IGNORECASE), "movement"),
]


def _is_non_feature_label(label: str) -> bool:
    stripped = (label or "").strip()
    if not stripped:
        return True
    return any(pattern.search(stripped) for pattern in _NON_FEATURE_LABEL_PATTERNS)


def _is_non_feature_text(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    return any(pattern.search(stripped) for pattern in _NON_FEATURE_LABEL_PATTERNS)


def _match_canonical_feature_key(label: str) -> Optional[str]:
    return _match_canonical_feature_key_with_value(label, None)


_COLOR_TOKENS = {
    "black",
    "white",
    "silver",
    "grey",
    "gray",
    "blue",
    "navy",
    "green",
    "red",
    "burgundy",
    "orange",
    "yellow",
    "gold",
    "rose",
    "champagne",
    "brown",
    "beige",
    "cream",
    "ivory",
    "pink",
    "purple",
}

_MATERIAL_TOKENS = {
    "stainless",
    "steel",
    "titanium",
    "ceramic",
    "gold",
    "bronze",
    "platinum",
    "carbon",
    "aluminium",
    "aluminum",
    "brass",
    "pvd",
    "dlc",
    "sapphire",
    "mineral",
    "acrylic",
    "hesalite",
    "leather",
    "rubber",
    "silicone",
    "nylon",
    "textile",
    "canvas",
    "alligator",
    "crocodile",
    "croc",
}

_BEZEL_TYPE_TOKENS = {"unidirectional", "uni-directional", "bidirectional", "bi-directional", "fixed", "rotating"}
_BEZEL_FUNCTION_TOKENS = {"tachymeter", "gmt", "countdown", "diving", "compass", "telemeter", "pulsometer", "slide rule"}
_DIAL_FINISH_TOKENS = {"sunray", "guilloche", "matte", "satin", "brushed", "lacquer", "sandblasted"}
_DIAL_TYPE_TOKENS = {"skeleton", "skeletonized", "open heart", "open-heart", "openheart"}


def _value_contains_any(value: str, tokens: set[str]) -> bool:
    text = _safe_str(value).lower()
    return any(re.search(rf"\b{re.escape(token)}\b", text) for token in tokens)


def _match_canonical_feature_key_with_value(label: str, value: Optional[str]) -> Optional[str]:
    normalized = _normalise_feature_label(label)
    if not normalized:
        return None
    candidates = [normalized, _strip_feature_label_unit_suffix(normalized)]
    for candidate in candidates:
        if candidate in _CANONICAL_LABEL_ALIASES:
            return _CANONICAL_LABEL_ALIASES[candidate]
    for pattern, key in _CANONICAL_LABEL_REGEX_RULES:
        for candidate in candidates:
            if candidate and pattern.search(candidate):
                return key
    value_text = _safe_str(value)
    if normalized in {"dial"} and value_text:
        if _value_contains_any(value_text, _MATERIAL_TOKENS):
            return "dial_material"
        if _value_contains_any(value_text, _DIAL_FINISH_TOKENS):
            return "dial_finish"
        if _value_contains_any(value_text, _DIAL_TYPE_TOKENS):
            return "dial_type"
        if _value_contains_any(value_text, _COLOR_TOKENS):
            return "dial_color"
        return "dial_color"
    if normalized in {"bezel"} and value_text:
        if _value_contains_any(value_text, _MATERIAL_TOKENS):
            return "bezel_material"
        if _value_contains_any(value_text, _BEZEL_FUNCTION_TOKENS):
            return "bezel_function"
        if _value_contains_any(value_text, _BEZEL_TYPE_TOKENS):
            return "bezel_type"
        if _value_contains_any(value_text, _COLOR_TOKENS):
            return "bezel_color"
        return "bezel_type"
    if normalized in {"case"} and value_text:
        if _value_contains_any(value_text, _MATERIAL_TOKENS):
            return "case_material"
        if re.search(r"\b\d+(?:\.\d+)?\s*mm\b", value_text.lower()):
            return "case_diameter"
        return "case_material"
    if normalized in {"crown"} and value_text:
        if _value_contains_any(value_text, _MATERIAL_TOKENS):
            return "crown_material"
        return "crown_type"
    if normalized in {"caseback", "case back"} and value_text:
        if _value_contains_any(value_text, _MATERIAL_TOKENS):
            return "caseback_material"
        return "caseback_type"
    if normalized in {"clasp"} and value_text:
        if _value_contains_any(value_text, _MATERIAL_TOKENS):
            return "clasp_material"
        return "clasp_type"
    return None


def _merge_feature_values(existing: str, incoming: str) -> str:
    existing = (existing or "").strip()
    incoming = (incoming or "").strip()
    if not existing:
        return incoming
    if not incoming:
        return existing
    if incoming.lower() in existing.lower():
        return existing
    if existing.lower() in incoming.lower():
        return incoming
    return f"{existing}; {incoming}"


_COMPOSITE_VALUE_SPLIT_RE = re.compile(r"[;|\n]+")


def _split_composite_value_tokens(value: str) -> List[str]:
    """Split a free-form spec value into semi-structured tokens."""
    text = _safe_str(value)
    if not text:
        return []
    for bullet in ("•", "·", "●", "▪", "◦"):
        text = text.replace(bullet, ";")
    parts = [part.strip() for part in _COMPOSITE_VALUE_SPLIT_RE.split(text) if part.strip()]
    return [re.sub(r"\s+", " ", part).strip(" -") for part in parts if part.strip(" -")]


_JEWELS_RE = re.compile(r"(?i)\b(\d{1,2})\s*jewels?\b")
_POWER_RESERVE_RE = re.compile(
    r"(?i)\b(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hours?)\b[^a-z0-9]{0,10}\bpower\s*reserve\b"
)
_POWER_RESERVE_RE_ALT = re.compile(
    r"(?i)\bpower\s*reserve\b[^0-9]{0,20}(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hours?)\b"
)
_FREQUENCY_VPH_RE = re.compile(r"(?i)\b(\d[\d, ]{2,})\s*vph\b")
_FREQUENCY_HZ_RE = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*hz\b")


def _extract_features_from_composite_tokens(tokens: Sequence[str]) -> tuple[Dict[str, str], List[str]]:
    """Extract canonical features from composite 'Functions/Complications' text.

    Returns: (derived_features, remaining_tokens)
    """
    derived: Dict[str, str] = {}
    remaining: List[str] = []

    for token in tokens:
        text = _safe_str(token)
        if not text or _is_non_feature_text(text):
            continue
        lowered = text.lower()
        mapped_any = False

        # Edition tokens frequently leak into Functions. Pull them out.
        if "edition" in lowered:
            edition_type, edition_number = _split_edition_type(text)
            if edition_type:
                derived["edition_type"] = _merge_feature_values(derived.get("edition_type", ""), edition_type)
                mapped_any = True
            if edition_number:
                derived["edition_number"] = _merge_feature_values(
                    derived.get("edition_number", ""), edition_number
                )
                mapped_any = True

        match = _JEWELS_RE.search(text)
        if match:
            derived["jewels"] = _merge_feature_values(derived.get("jewels", ""), match.group(1))
            mapped_any = True

        match = _POWER_RESERVE_RE.search(text) or _POWER_RESERVE_RE_ALT.search(text)
        if match:
            derived["power_reserve"] = _merge_feature_values(
                derived.get("power_reserve", ""), f"{match.group(1)} Hr"
            )
            mapped_any = True

        match = _FREQUENCY_VPH_RE.search(text)
        if match:
            digits = re.sub(r"[^0-9]+", "", match.group(1))
            if digits:
                derived["frequency"] = _merge_feature_values(derived.get("frequency", ""), f"{digits} vph")
                mapped_any = True

        match = _FREQUENCY_HZ_RE.search(text)
        if match:
            derived["frequency"] = _merge_feature_values(derived.get("frequency", ""), f"{match.group(1)} Hz")
            mapped_any = True

        if "buckle" in lowered:
            derived["buckle_type"] = _merge_feature_values(derived.get("buckle_type", ""), text)
            mapped_any = True

        if "clasp" in lowered:
            derived["clasp_type"] = _merge_feature_values(derived.get("clasp_type", ""), text)
            mapped_any = True

        if any(word in lowered for word in ("sapphire", "mineral", "hesalite", "acrylic", "crystal", "glass")):
            derived["crystal_material"] = _merge_feature_values(derived.get("crystal_material", ""), text)
            mapped_any = True

        if "luminova" in lowered or re.search(r"\blume\b", lowered):
            derived["lume"] = _merge_feature_values(derived.get("lume", ""), text)
            mapped_any = True

        is_strap_hardware = any(word in lowered for word in ("buckle", "clasp", "strap", "bracelet"))
        has_coating_word = "coating" in lowered or "coated" in lowered
        has_strong_coating_token = any(word in lowered for word in ("dlc", "pvd", "diamond-like carbon", "diamond like carbon"))
        has_ip_context = "ip" in lowered and (has_coating_word or "case" in lowered)
        if not is_strap_hardware and (has_coating_word or has_strong_coating_token or has_ip_context):
            # Heuristic: AR/anti-reflective is typically a crystal coating, not a case coating.
            if re.search(r"(?i)\banti[- ]?reflect|\\bar\\b", text):
                derived["crystal_coating"] = _merge_feature_values(derived.get("crystal_coating", ""), text)
            else:
                derived["case_coating"] = _merge_feature_values(derived.get("case_coating", ""), text)
            mapped_any = True

        if not mapped_any:
            remaining.append(text)

    return derived, _dedupe_preserve(remaining)


_CALIBER_MARKER_RE = re.compile(r"(?i)\bcalib(?:re|er)\b")
_CALIBER_CODE_RE = re.compile(
    r"\b(?:[A-Z]{1,4}\d{1,5}(?:[A-Z])?(?:[-/]\d+)?|[A-Z]{1,4}\d{1,3}\.\d{1,3}|\d{3,4}(?:[-/]\d+)?)\b"
)
_MOVEMENT_TYPE_NOISE_RE = re.compile(
    r"(?i)\b(automatic|self-winding|self winding|manual|hand-wound|hand wound|quartz|solar|hybrid|kinetic|eco-drive|spring drive)\b"
)
_CALIBER_VALUE_NOISE_RE = re.compile(
    r"(?i)\b(?:\d[\d, ]*\s*vph|\d+(?:\.\d+)?\s*hz|\d+(?:\.\d+)?\s*(?:h|hr|hrs|hours?)\b|\d{1,2}\s*jewels?\b)\b"
)


def _clean_caliber_value(value: str) -> str:
    text = _safe_str(value)
    if not text:
        return ""
    candidate = text
    candidate = _MOVEMENT_TYPE_NOISE_RE.sub("", candidate)
    candidate = re.sub(r"(?i)\bmovement\b", "", candidate)
    candidate = _CALIBER_VALUE_NOISE_RE.sub("", candidate)
    candidate = re.sub(r"(?i)\bpower\s*reserve\b", "", candidate)
    candidate = re.sub(r"\(\s*\)", "", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" -()")
    return candidate


def _extract_caliber_from_movement_value(value: str) -> str:
    """Extract a caliber-like string from a movement value without polluting movement type."""
    text = _safe_str(value)
    if not text:
        return ""
    if not (_CALIBER_MARKER_RE.search(text) or _CALIBER_CODE_RE.search(text)):
        return ""
    candidate = _clean_caliber_value(text)
    if candidate and re.fullmatch(r"(?i)calib(?:re|er)", candidate):
        match = _CALIBER_CODE_RE.search(text)
        return match.group(0) if match else ""
    if not candidate:
        match = _CALIBER_CODE_RE.search(text)
        if match:
            return match.group(0)
    return candidate


def canonicalize_watch_feature_pairs(feature_pairs: Any) -> tuple[Dict[str, str], List[Dict[str, str]]]:
    """Map free-form spec key/value pairs into a canonical feature dict.

    Returns:
      (canonical_features, unmapped_pairs)
    """
    canonical: Dict[str, str] = {}
    unmapped: List[Dict[str, str]] = []

    if isinstance(feature_pairs, dict):
        feature_pairs = [{"label": k, "value": v} for k, v in feature_pairs.items()]
    if not isinstance(feature_pairs, list):
        return canonical, unmapped

    for item in feature_pairs:
        if not isinstance(item, dict):
            continue
        label = _safe_str(item.get("label") or item.get("name"))
        value = _safe_str(item.get("value"))
        if not label or not value:
            continue
        if _is_non_feature_label(label):
            continue
        key = _match_canonical_feature_key_with_value(label, value)
        if not key:
            unmapped.append({"label": label, "value": value})
            continue
        if key == "movement":
            movement_type = _normalize_movement_category(value)
            if movement_type:
                canonical["movement"] = _merge_feature_values(canonical.get("movement", ""), movement_type)
            caliber_candidate = _extract_caliber_from_movement_value(value)
            if caliber_candidate:
                canonical["caliber"] = _merge_feature_values(canonical.get("caliber", ""), caliber_candidate)
            continue
        if key == "caliber":
            movement_type = _normalize_movement_category(value)
            if movement_type:
                canonical["movement"] = _merge_feature_values(canonical.get("movement", ""), movement_type)
            cleaned = _clean_caliber_value(value)
            if cleaned:
                canonical["caliber"] = _merge_feature_values(canonical.get("caliber", ""), cleaned)
            continue
        if key in {"functions", "complications"}:
            tokens = _split_composite_value_tokens(value)
            derived, remaining = _extract_features_from_composite_tokens(tokens)
            for derived_key, derived_value in derived.items():
                canonical[derived_key] = _merge_feature_values(canonical.get(derived_key, ""), derived_value)
            if remaining:
                canonical[key] = _merge_feature_values(
                    canonical.get(key, ""), "; ".join(_dedupe_preserve(remaining))
                )
            continue
        canonical[key] = _merge_feature_values(canonical.get(key, ""), value)

    return canonical, unmapped


def build_metafield_header(label: str, key: str, namespace: str = "custom") -> str:
    safe_key = _slugify(key).replace("-", "_")
    safe_key = re.sub(r"[^a-z0-9_]+", "_", safe_key).strip("_")
    if not safe_key:
        safe_key = "unknown"
    return f"{label} (product.metafields.{namespace}.{safe_key})"


def _parse_first_number(text: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)", _safe_str(text))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _format_mm(value: float) -> str:
    if value.is_integer():
        return f"{int(value)}mm"
    return f"{value:g}mm"


def _normalise_dimension_mm(text: str, prefer_range: tuple[float, float]) -> str:
    """Normalise a dimension string to a single '<N>mm' token when possible."""
    raw = _safe_str(text)
    if not raw:
        return ""
    lowered = raw.lower()
    values = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*mm\b", lowered)]
    if not values:
        return ""
    lo, hi = prefer_range
    for candidate in values:
        if lo <= candidate <= hi:
            return _format_mm(candidate)
    return _format_mm(values[0])


def _extract_case_dimensions(value: str) -> tuple[str, str, str]:
    """Best-effort parsing of '40 x 42 x 11mm' style strings.

    Returns: (diameter_mm, lug_to_lug_mm, thickness_mm)
    """
    text = _safe_str(value).lower()
    if not text:
        return ("", "", "")
    match3 = re.search(
        r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*mm\b",
        text,
    )
    if match3:
        d, l2l, thick = (float(match3.group(1)), float(match3.group(2)), float(match3.group(3)))
        return (_format_mm(d), _format_mm(l2l), _format_mm(thick))
    match2 = re.search(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*mm\b", text)
    if match2:
        d, thick = (float(match2.group(1)), float(match2.group(2)))
        return (_format_mm(d), "", _format_mm(thick))
    return ("", "", "")


_COATING_TOKENS = (
    "dlc",
    "pvd",
    "ip",
    "diamond-like carbon",
    "diamond like carbon",
    "carbon coating",
    "coating",
    "coated",
)


def _split_case_material(value: str) -> tuple[str, str]:
    """Return (material, coating_details)."""
    text = _safe_str(value)
    if not text:
        return ("", "")

    lowered = text.lower()
    coating = ""
    if any(token in lowered for token in _COATING_TOKENS):
        # Try to capture a readable coating phrase for a dedicated metafield.
        match = re.search(
            r"(?i)\b(black\s+)?(dlc|pvd|ip|diamond[- ]like carbon)\b[^,;.)]*",
            text,
        )
        if match:
            coating = match.group(0).strip()
        else:
            coating = "Coated"

    # Keep the base material short and Shopify-friendly.
    material = text
    # Drop obvious case/coating suffixes.
    material = re.sub(r"(?i)\bcase\b", "", material).strip()
    material = re.sub(r"(?i)\bw/\b", "with", material).strip()
    if coating:
        material = re.sub(r"(?i)\bwith\b.*", "", material).strip()
    # Trim at the first delimiter when the string is too verbose.
    for sep in (",", "|", ";"):
        if sep in material:
            material = material.split(sep, 1)[0].strip()
            break
    material = re.sub(r"\s+", " ", material).strip(" -")
    return (material, coating)


def _split_strap_value(value: str) -> tuple[str, str]:
    """Return (strap_clean, strap_notes)."""
    text = _safe_str(value)
    if not text:
        return ("", "")
    notes: List[str] = []
    cleaned = text
    # Pull out parenthetical availability notes.
    for match in re.finditer(r"\(([^)]{0,200})\)", text):
        inner = (match.group(1) or "").strip()
        if not inner:
            continue
        if re.search(r"(?i)\bavailable\b|\balso\b|\bother\b", inner):
            notes.append(inner)
    cleaned = re.sub(r"\([^)]{0,200}\)", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    return (cleaned, "; ".join(_dedupe_preserve(notes)))


def _split_edition_type(value: str) -> tuple[str, str]:
    """Return (edition_type, edition_number_or_range)."""
    text = _safe_str(value)
    if not text:
        return ("", "")
    lowered = text.lower()
    edition_type = ""
    if "limited" in lowered and "edition" in lowered:
        edition_type = "Limited Edition"
    elif "special" in lowered and "edition" in lowered:
        edition_type = "Special Edition"
    else:
        # Conservative fallback: keep the first chunk.
        edition_type = re.split(r"[|(/]", text, 1)[0].strip()

    number = ""
    # Common patterns: "1 - 1907", "1-50", "12/500", "No. 12 of 500"
    for pattern in (
        r"\b\d+\s*[-–]\s*\d+\b",
        r"\b\d+\s*/\s*\d+\b",
        r"\bno\.\s*\d+\b",
        r"\b\d+\s+of\s+\d+\b",
    ):
        match = re.search(pattern, lowered, re.IGNORECASE)
        if match:
            number = match.group(0).strip()
            break
    return (edition_type, number)


def _extract_grams_from_text(value: Any) -> str:
    text = _safe_str(value).lower()
    if not text:
        return ""
    match_g = re.search(r"(\d+(?:\.\d+)?)\s*g\b", text)
    if match_g:
        return match_g.group(1)
    match_kg = re.search(r"(\d+(?:\.\d+)?)\s*kg\b", text)
    if match_kg:
        try:
            kg = float(match_kg.group(1))
        except ValueError:
            return ""
        grams = kg * 1000.0
        return f"{grams:g}"
    return ""


def _enrich_watch_features(row: Dict[str, str], canonical_features: Dict[str, str]) -> None:
    """Make the 'Watch features' metafield more stable by supplementing it with canonical specs."""
    header = "Watch features (product.metafields.custom.watch_features)"
    existing = row.get(header, "") or ""
    tokens: List[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        cleaned = token.strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        tokens.append(cleaned)

    for token in existing.split(";"):
        _add(token)

    # Canonical spec-driven hints.
    crystal = canonical_features.get("crystal_material", "")
    if "sapphire" in crystal.lower():
        _add("sapphire crystal")

    coating = row.get(build_metafield_header("Case Coating", "case_coating"), "") or ""
    if "dlc" in coating.lower():
        _add("dlc coating")
    elif "pvd" in coating.lower():
        _add("pvd coating")

    power_reserve = canonical_features.get("power_reserve", "")
    if power_reserve:
        _add("power reserve")

    jewels = canonical_features.get("jewels", "")
    if jewels:
        number = _parse_first_number(jewels)
        if number is not None:
            _add(f"{int(number)} jewels")
        else:
            _add("jewels")

    complications = canonical_features.get("complications", "")
    for part in complications.split(";"):
        _add(part)

    movement = row.get("Movement (product.metafields.custom.movement)", "")
    if movement:
        _add(movement.lower())

    # Only write back if we changed anything.
    enriched = "; ".join(tokens)
    if enriched.strip() and enriched.strip() != existing.strip():
        row[header] = enriched


def _format_header_row(headers: Sequence[str]) -> str:
    # Join headers into one line so the prompt shows the exact CSV order.
    return ",".join(headers)


def _format_field_guidance(headers: Sequence[str]) -> str:
    # Build a bullet list that pairs each header with its policy and description.
    lines: List[str] = []
    for name in headers:
        info = SHOPIFY_FIELD_GUIDANCE.get(name)
        if not info:
            continue
        policy_key = info.get("policy", "")
        label = FIELD_POLICY_LABELS.get(policy_key, policy_key.upper())
        description = info.get("description", "").strip()
        lines.append(f"- {name} [{label}]: {description}")
    return "\n".join(lines)


# Rotate user agents to look more like normal shoppers.
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]

REQUEST_TIMEOUT = 60
MAX_PROMPT_HTML_CHARS = 40_000
MAX_PROMPT_TEXT_CHARS = 12_000
MAX_IMAGE_CANDIDATES = 12
MAX_VARIANTS = 12  # Shopify CSV rows per product variant; extend if needed



# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PagePayload:
    # Bundle of cleaned page info that feeds the AI prompt.
    url: str
    html: str
    text: str
    image_urls: List[str]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _slug_from_url(url: str) -> str:
    # Use the last URL path piece as a simple handle fallback.
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    return parts[-1] if parts else re.sub(r"[^a-z0-9]+", "-", parsed.netloc.lower())


def _slugify(value: Any) -> str:
    text = _safe_str(value).lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def _canonicalize_product_url(url: str) -> str:
    # Strip query strings and fragments so variant links collapse to one handle.
    parsed = urlparse(url)
    canonical = parsed._replace(query="", fragment="")
    # Normalise trailing slashes so https://example.com/product and .../product/ match.
    path = canonical.path or ""
    if len(path) > 1 and path.endswith("/"):
        canonical = canonical._replace(path=path.rstrip("/"))
    return urlunparse(canonical)


def _canonicalize_image_url(url: str) -> str:
    """Remove common resizing params so Shopify/CDN images stay stable."""
    raw = _safe_str(url)
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except Exception:
        return raw
    if not parsed.query:
        return raw
    drop_params = {"width", "height", "format", "quality", "crop", "pad_color", "vpb"}
    kept = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() not in drop_params]
    rebuilt = parsed._replace(query=urlencode(kept))
    return urlunparse(rebuilt)


def _clean_html(html: str, base_url: str) -> PagePayload:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select(
        "button, nav, form, .cookie-banner, .popup, .related-products, .shipping-info, .size-guide"
    ):
        tag.decompose()
    # Drop styles/noscript outright, but keep data-rich scripts for the model.
    for tag in soup(["style", "noscript"]):
        tag.decompose()
    preserved_script_types = {"application/ld+json", "text/x-magento-init"}
    preserved_script_keywords = ("wpmdatlayer", "datalayer", "shopify.currency", "product:")
    for script in soup.find_all("script"):
        script_type = (script.get("type") or "").strip().lower()
        if script_type in preserved_script_types:
            continue
        script_text = script.get_text(" ", strip=False) or ""
        if script_text and any(keyword in script_text.lower() for keyword in preserved_script_keywords):
            continue
        script.decompose()
    # Collapse whitespace for a tidy text block the model can read.
    text = " ".join(soup.get_text(" ", strip=True).split())
    # Save absolute image links so the prompt can mention them.
    image_urls = _extract_image_sources(soup, base_url)
    serialized = str(soup)
    return PagePayload(url="", html=serialized, text=text, image_urls=image_urls)


def _extract_image_sources(soup: BeautifulSoup, base_url: str) -> List[str]:
    images: List[str] = []
    seen: set[str] = set()
    # Walk through every <img> and normalise its URL.
    for img in soup.select("img[src]"):
        src = (img.get("data-src") or img.get("src") or "").strip()
        if not src:
            continue
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = urljoin(base_url, src)
        src = _canonicalize_image_url(src)
        # Avoid duplicates so the prompt stays short.
        if src and src not in seen:
            seen.add(src)
            images.append(src)
        if len(images) >= MAX_IMAGE_CANDIDATES:
            break
    return images


def _extract_currency_from_html(html: str) -> str:
    """Best-effort extraction of the store's default currency as a 3-letter code.

    Tries common Shopify and schema.org patterns, returning an uppercase
    ISO-4217-like code (e.g., USD, CHF) or an empty string if not found.
    """
    # Prefer Shopify's declared shop/default currency, which remains stable even
    # when the storefront localises prices for the visitor.
    shopify_currency_match = re.search(r'Shopify\.currency\s*=\s*\{([^}]+)\}', html)
    if shopify_currency_match:
        blob = shopify_currency_match.group(1)
        for key in ("shopCurrency", "defaultCurrency"):
            m = re.search(rf'"{key}"\s*:\s*"([A-Za-z]{{3}})"', blob)
            if m:
                return m.group(1).upper()
        m_active = re.search(r'"active"\s*:\s*"([A-Za-z]{3})"', blob)
        if m_active:
            return m_active.group(1).upper()

    # Otherwise, inspect structured data that may encode the canonical currency.
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return ""

    # Structured data blocks sometimes repeat the currency label.
    # JSON-LD blocks may contain priceCurrency
    for script in soup.select("script[type='application/ld+json']"):
        data = script.string or ""
        if not data:
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue

        def _find_currency(obj: Any) -> str:
            if isinstance(obj, dict):
                # Direct field
                cur = obj.get("priceCurrency")
                if isinstance(cur, str) and re.fullmatch(r"[A-Za-z]{3}", cur.strip()):
                    return cur.strip().upper()
                # Nested search
                for v in obj.values():
                    found = _find_currency(v)
                    if found:
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = _find_currency(item)
                    if found:
                        return found
            return ""

        code = _find_currency(payload)
        if code:
            return code

    # As a last resort read the meta tags for a 3-letter code.
    node = soup.select_one("meta[itemprop='priceCurrency'], meta[property='product:price:currency']")
    if node:
        content = (node.get("content") or "").strip()
        if re.fullmatch(r"[A-Za-z]{3}", content):
            return content.upper()

    return ""


def _extract_grams_from_html(html: str) -> str:
    """Best-effort extraction of weight from the full HTML."""
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return ""
    text = " ".join(soup.get_text(" ", strip=True).split())
    if not text:
        return ""
    match = re.search(r"(?i)\bweight\b[^0-9]{0,40}(\d+(?:\.\d+)?)\s*(kg|g)\b", text)
    if not match:
        return ""
    value = match.group(1)
    unit = match.group(2).lower()
    try:
        numeric = float(value)
    except ValueError:
        return ""
    if unit == "kg":
        numeric *= 1000.0
    if numeric <= 0:
        return ""
    return f"{numeric:g}"


def _extract_product_links_from_soup(soup: BeautifulSoup, base_url: str) -> List[str]:
    anchors = soup.select("a[href]")
    candidates: List[str] = []
    seen_canonical: set[str] = set()
    base_host = urlparse(base_url).netloc
    # Walk every anchor and keep only the ones that look like watch pages.
    for anchor in anchors:
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript")):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        if parsed.netloc and base_host and parsed.netloc != base_host:
            continue

        path = (parsed.path or "").strip("/")
        path_lower = (parsed.path or "").lower()
        if not path:
            continue
        segments = path.split("/")
        last_segment = segments[-1]
        last_no_ext = last_segment.split(".", 1)[0].lower()

        # Skip Shopify marketing pages (/pages/...) and pure collection links that
        # don't point to a specific /products/ slug, unless they end with .html.
        contains_products_segment = "/products/" in path_lower
        contains_pages_segment = "/pages/" in path_lower
        contains_collection_only = "/collections/" in path_lower and not contains_products_segment
        if (contains_pages_segment or contains_collection_only) and not path_lower.endswith(".html"):
            continue

        href_lower = href.lower()
        text_lower = (anchor.get_text(" ") or "").lower()

        # Strong Shopify-style product signals.
        is_shopify_product = "/products/" in parsed.path.lower() or "/product/" in parsed.path.lower()

        # Generic catalog/utility slugs we want to avoid treating as products.
        catalog_like_slugs = {
            "all",
            "all-watches",
            "watches",
            "watch",
            "filter-by",
            "collections",
            "collection",
            "blog",
            "stories",
            "story",
            "about",
            "contact",
            "service",
            "support",
            "search",
            "account",
            "login",
            "register",
            "cart",
            "wishlist",
            "manuals",
            "watch-manuals",
            "privacy-policy",
            "terms",
            "terms-of-service",
            "faq",
            "index",
            "home",
        }
        catalog_like_substrings = ("collection", "category", "filter", "watches", "watch-manuals", "manuals")
        is_catalog_like = last_no_ext in catalog_like_slugs or any(
            sub in last_no_ext for sub in catalog_like_substrings
        )

        # Original loose heuristics based on URL/text containing watch-related tokens.
        is_watch_link = any(token in href_lower for token in ("/product", "/watch")) or "variant" in href_lower
        mentions_watch = any(token in text_lower for token in ("watch", "watches", "product", "shop"))

        # Generic product slug heuristic: longer, slug-like path segment with hyphens or digits.
        looks_like_product_slug = (
            not is_catalog_like
            and len(last_no_ext) >= 4
            and any(ch.isalpha() for ch in last_no_ext)
            and ("-" in last_no_ext or any(ch.isdigit() for ch in last_no_ext))
        )

        if is_shopify_product or is_watch_link or looks_like_product_slug:
            canonical = _canonicalize_product_url(absolute)
            if canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
            candidates.append(canonical)
    return candidates


def _looks_like_single_product_page(html: str) -> bool:
    """Heuristic check to distinguish single-product pages from catalogs/landing pages.

    This does not depend on the model and helps avoid false negatives when the
    AI misclassifies a genuine product page as non-product.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return False

    text = soup.get_text(" ", strip=True).lower()
    if any(token in text for token in ("add to cart", "add to bag", "add to basket")):
        return True

    meta = soup.select_one("meta[property='og:type'], meta[name='og:type']")
    if meta:
        content = (meta.get("content") or "").lower()
        if "product" in content:
            return True

    for script in soup.select("script[type='application/ld+json']"):
        data = script.string or ""
        if not data:
            continue
        if '"@type"' in data and "product" in data.lower():
            return True

    return False


def _discover_next_catalog_urls(soup: BeautifulSoup, base_url: str) -> List[str]:
    next_links: List[str] = []
    seen: set[str] = set()

    def _add(url: str) -> None:
        # Guard against blank or duplicate pagination links.
        url = url.strip()
        if not url or url in seen:
            return
        seen.add(url)
        next_links.append(url)

    # Start with explicit rel="next" hints.
    for tag in soup.select('link[rel="next"], a[rel="next"]'):
        href = tag.get("href") or ""
        if href:
            _add(urljoin(base_url, href))

    candidate_selectors = [
        "a.next",
        "a.next-page",
        "a.pagination__next",
        "li.next a",
        "a[aria-label*='Next']",
        "button.next",
        "a[title*='Next']",
    ]
    # Check common pagination classes used by Shopify themes.
    for selector in candidate_selectors:
        for tag in soup.select(selector):
            href = tag.get("href") or ""
            if href:
                _add(urljoin(base_url, href))

    # Finally, read every anchor for words like "next" or "more".
    for anchor in soup.select("a[href]"):
        text = (anchor.get_text(" ") or "").strip().lower()
        href = (anchor.get("href") or "").strip()
        if not text or not href:
            continue
        if "next" in text or "older" in text or "more" in text:
            _add(urljoin(base_url, href))
    return next_links


def _truncate(value: str, limit: int) -> str:
    # Shorten long text while leaving a hint that data was trimmed.
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]"


def _pick_user_agent() -> str:
    # Randomise the User-Agent to look less like a bot.
    return random.choice(USER_AGENTS)


def _default_headers() -> Dict[str, str]:
    # Browser-mimicking headers that work across both requests and Playwright.
    return {
        "User-Agent": _pick_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }


def _build_session() -> requests.Session:
    # Use a shared session so headers and cookies persist across requests.
    session = requests.Session()
    session.headers.update(_default_headers())
    return session


@contextmanager
def _playwright_context():
    # Provide a shared Playwright browser context when available.
    if sync_playwright is None:  # pragma: no cover - optional dependency
        yield None
        return
    browser = None
    context = None
    yielded = False
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            headers = _default_headers()
            user_agent = headers.pop("User-Agent", _pick_user_agent())
            context = browser.new_context(user_agent=user_agent, extra_http_headers=headers)
            yielded = True
            yield context
    except Exception as exc:  # pragma: no cover - runtime safety
        if yielded:
            raise
        print(f"  ! Playwright failed to start; falling back to requests-only mode: {exc}")
        yield None
    finally:
        try:
            if context is not None:
                context.close()
            if browser is not None:
                browser.close()
        except Exception:
            pass


def fetch_html(session: requests.Session, url: str, browser_context: Any = None) -> str:
    # Prefer a rendered DOM via Playwright, then fall back to raw HTTP.
    if browser_context is not None:
        page = None
        try:
            page = browser_context.new_page()
            page.goto(url, wait_until="networkidle", timeout=REQUEST_TIMEOUT * 1000)
            html = page.content()
            page.close()
            return html
        except PlaywrightTimeoutError as exc:  # pragma: no cover - runtime safety
            print(f"  ! Playwright timed out for {url}: {exc}; falling back to requests")
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"  ! Playwright failed for {url}: {exc}; falling back to requests")
        finally:
            try:
                if page is not None and not page.is_closed():
                    page.close()
            except Exception:
                pass
    # Fetch a page and raise clear, user-friendly errors when the site fails to respond.
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as exc:
        raise RuntimeError(
            f"Timed out fetching {url} after {REQUEST_TIMEOUT} seconds. "
            "This often means the website's firewall or bot protection is blocking automated requests."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Network error while fetching {url}: {exc}") from exc

    status = resp.status_code
    snippet = (resp.text or "")[:4096].lower()
    if status == 403 and ("just a moment" in snippet or "access denied" in snippet or "forbidden" in snippet):
        raise RuntimeError(
            f"Blocked by website firewall/bot protection when fetching {url} (HTTP {status}). "
            "The server returned a challenge/denied page instead of the catalog HTML."
        )

    resp.raise_for_status()
    return resp.text


def discover_product_links(
    session: requests.Session,
    catalog_url: str,
    limit: int,
    browser_context: Any = None,
) -> List[str]:
    queue: List[str] = [catalog_url]
    visited_pages: set[str] = set()
    discovered: List[str] = []

    def _limit_reached() -> bool:
        return limit > 0 and len(discovered) >= limit

    # Breadth-first crawl through catalog pages until we hit the limit.
    while queue and not _limit_reached():
        current_url = queue.pop(0)
        if current_url in visited_pages:
            continue
        visited_pages.add(current_url)
        try:
            html = fetch_html(session, current_url, browser_context=browser_context)
        except Exception as exc:  # pragma: no cover - network/runtime safety
            print(f"  ! Failed to fetch catalog page {current_url}: {exc}")
            # If we cannot even fetch the root catalog URL (e.g., blocked by WAF),
            # bubble the error up so the caller can show a clear message instead
            # of a generic "No product URLs discovered" notice.
            if current_url == catalog_url:
                raise
            continue
        soup = BeautifulSoup(html, "html.parser")
        product_links = _extract_product_links_from_soup(soup, current_url)
        # Add new product URLs in the order we find them.
        for product_url in product_links:
            if product_url in discovered:
                continue
            discovered.append(product_url)
            if _limit_reached():
                break
        # Enqueue pagination links for further crawling.
        for next_page in _discover_next_catalog_urls(soup, current_url):
            if next_page not in visited_pages and next_page not in queue:
                queue.append(next_page)
    if limit > 0:
        return discovered[:limit]
    return discovered


def _candidate_catalog_urls_from_soup(soup: BeautifulSoup, base_url: str, product_url: str) -> List[str]:
    anchors = soup.select("a[href]")
    candidates: List[str] = []
    seen: set[str] = set()

    parsed_product = urlparse(product_url)
    product_path = parsed_product.path
    product_last_segment = [p for p in product_path.split("/") if p][-1] if product_path else ""

    keywords = (
        "all watches",
        "all products",
        "watches",
        "watch",
        "collection",
        "collections",
        "timepieces",
        "shop",
        "filter-by",
    )

    for anchor in anchors:
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.netloc != parsed_product.netloc:
            continue
        if absolute == product_url:
            continue
        path = parsed.path or ""
        last_segment = [p for p in path.split("/") if p][-1] if path else ""
        if last_segment == product_last_segment:
            continue
        text = (anchor.get_text(" ") or "").strip().lower()
        haystack = f"{text} {path.lower()}"
        if any(kw in haystack for kw in keywords):
            if absolute not in seen:
                seen.add(absolute)
                candidates.append(absolute)
    return candidates


def discover_catalog_from_product(
    session: requests.Session,
    product_url: str,
    max_products: int,
    browser_context: Any = None,
) -> List[str]:
    # Use a single product page as a seed to locate a broader catalog/collection.
    try:
        html = fetch_html(session, product_url, browser_context=browser_context)
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"  ! Failed to fetch seed product page {product_url}: {exc}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    base_url = product_url

    candidates = _candidate_catalog_urls_from_soup(soup, base_url, product_url)
    # As a last resort, also consider the site root as a candidate.
    parsed = urlparse(product_url)
    site_root = f"{parsed.scheme}://{parsed.netloc}/"
    if site_root not in candidates:
        candidates.append(site_root)

    best_links: List[str] = []
    best_source: Optional[str] = None
    limit = max(0, max_products or 0)

    for candidate in candidates:
        discovery_limit = limit or 0
        links = discover_product_links(
            session=session,
            catalog_url=candidate,
            limit=discovery_limit,
            browser_context=browser_context,
        )
        if len(links) > len(best_links):
            best_links = links
            best_source = candidate
        if limit > 0 and len(best_links) >= limit:
            break

    if best_source:
        print(f"[catalog-discovery] Seed product {product_url} → catalog {best_source} ({len(best_links)} products)")
    return best_links


def build_prompt(headers: Sequence[str], payload: PagePayload) -> str:
    # Compose the long-form instructions sent to the AI model.
    header_row = _format_header_row(headers)
    field_guidance = _format_field_guidance(headers)
    policy_legend = (
        "[FILL] = actively capture from catalog/product pages whenever the information exists "
        "(HTML specs, descriptions, structured data).\n"
        "[OPPORTUNISTIC] = only populate when the storefront explicitly exposes a confident value "
        "(JSON-LD, data-* attributes, clearly labeled UI).\n"
        "[STATIC] = leave blank so importer defaults remain unless the storefront publishes an "
        "authoritative admin value."
    )
    field_spec = "\n".join(f"- {name}: {desc}" for name, desc in RAW_FIELD_SPECS.items())
    prompt = f"""
    You are a content sanitizer and expert watch merchandiser. Extract clean,
    brand-agnostic product facts from the provided page and return them as RAW
    data (not Shopify-ready).

    When generating 'body_html', strictly exclude non-product UI elements.
    Specifically remove phrases like "Shop now", "View more", "Download Size Guide",
    "Delivery time", and inputs like "Please select a wrist size". Return ONLY
    descriptive product content and specifications in clean HTML.

    Output requirements:
    - Respond with ONE JSON object whose top-level keys exactly match the fields
      described below.
    - Do NOT attempt to format to Shopify column headers. Only return the raw
      entities; the caller will handle mapping.
    - Use empty strings, empty arrays, or false when the source material is
      missing, contradictory, or ambiguous. Never guess.
    - Preserve evidence faithfully. If multiple values conflict, prefer the one
      most emphasized on the page and leave a note in other_features.
    - Limit lists (tags, images, features) to the strongest matches ordered by
      relevance.
    - For `feature_pairs`, ONLY emit explicit watch specification pairs
      (label/value) from spec tables, definition lists, or clearly labeled spec
      sections. Exclude shipping/returns/warranty/payment and other non-watch UI.
      Do not invent new labels; use the page's label text.

    Shopify CSV header order for ai_watch_scraper.csv (keep this EXACT order):
    {header_row}

    Field policy legend:
    {policy_legend}

    Column-by-column guidance:
    {field_guidance}

    Raw field dictionary you must return in the JSON (ensure every key exists):
    {field_spec}

    Input context:
    - Product URL: {payload.url}
    - Observed image URLs: {json.dumps(payload.image_urls, ensure_ascii=False)}

    Cleaned page text (truncated):
    ```
    {_truncate(payload.text, MAX_PROMPT_TEXT_CHARS)}
    ```

    HTML snippet (truncated):
    ```html
    {_truncate(payload.html, MAX_PROMPT_HTML_CHARS)}
    ```

    Respond with ONLY valid JSON (no code fences, no explanations).
    """
    return textwrap.dedent(prompt).strip()


def configure_model() -> genai.GenerativeModel:
    if not GOOGLE_API_KEY:
        raise SystemExit(
            "Missing Google API key. Set GOOGLE_API_KEY env var or populate HARDCODED_GOOGLE_API_KEY."
        )
    # Register the API key once so every later call is authenticated.
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config={
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 32,
        },
    )


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    # Some model responses wrap JSON in ```json fences; peel them off.
    fence_match = re.match(r"```(?:json)?\s*(.*)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def invoke_model(model: genai.GenerativeModel, prompt: str, retries: int = 2) -> Dict[str, Any]:
    last_error: Optional[str] = None
    message = prompt
    # Retry a few times because the model may reply with malformed JSON.
    for attempt in range(1, retries + 2):
        response = model.generate_content(message)
        text = response.text or ""
        cleaned = _strip_json_fences(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            last_error = f"Attempt {attempt}: {exc}"
            message = (
                prompt
                + "\n\n"
                + "Reminder: respond with ONLY valid JSON matching the requested columns. "
                + "Do not include code fences or explanations."
            )
    raise ValueError(last_error or "Model failed to return valid JSON")


# ---------------------------------------------------------------------------
# Row normalisation
# ---------------------------------------------------------------------------


def _normalise_numeric(value: str) -> str:
    # Remove currency symbols or stray text so only numbers remain.
    if value is None:
        return ""
    value = str(value).strip()
    if not value:
        return ""
    cleaned = re.sub(r"[^0-9.\-]", "", value)
    return cleaned


def _safe_str(value: Any) -> str:
    # Convert any input to a trimmed string, falling back to blank.
    if value is None:
        return ""
    return str(value).strip()


def _clean_list(value: Any) -> List[str]:
    # Ensure we always return a simple list of strings.
    if value is None:
        return []
    if isinstance(value, str):
        tokens = re.split(r"[;,]", value)
        return [token.strip() for token in tokens if token.strip()]
    if isinstance(value, (list, tuple, set)):
        result: List[str] = []
        for item in value:
            text = _safe_str(item)
            if text:
                result.append(text)
        return result
    return []


def _dedupe_preserve(items: Sequence[str]) -> List[str]:
    # Keep the first appearance of each string and drop repeats.
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _clean_image_list(value: Any) -> List[str]:
    # Build a unique list of image URLs regardless of the incoming format.
    result: List[str] = []
    if isinstance(value, list):
        for item in value:
            url = _canonicalize_image_url(_safe_str(item))
            if url and url not in result:
                result.append(url)
    elif isinstance(value, str):
        url = _canonicalize_image_url(_safe_str(value))
        if url:
            result.append(url)
    return result


def _coerce_bool(value: Any) -> str:
    # Normalise truthy strings into Shopify's lowercase true/false.
    if isinstance(value, bool):
        return "true" if value else "false"
    text = _safe_str(value).lower()
    if text in {"true", "yes", "y", "1"}:
        return "true"
    if text in {"false", "no", "n", "0"}:
        return "false"
    return ""


def _normalize_target_gender(value: Any) -> str:
    text = _safe_str(value).lower()
    if not text:
        return ""
    if re.search(r"\bunisex\b", text):
        return "Unisex"
    if re.search(r"\b(women|womens|woman|female|ladies|lady)\b", text):
        return "Women"
    if re.search(r"\b(men|mens|man|male)\b", text):
        return "Men"
    return ""


def _normalize_age_group(value: Any) -> str:
    text = _safe_str(value).lower()
    if not text:
        return ""
    if re.search(r"\b(adult|adults)\b", text):
        return "adult"
    if re.search(r"\b(child|children|kid|kids|youth|junior|boys|girls)\b", text):
        return "child"
    return ""


def _simplify_color(value: Any) -> str:
    text = _safe_str(value)
    if not text:
        return ""
    text = re.sub(r"\([^)]*\)", "", text)
    for sep in ("/", ",", ";", "|"):
        if sep in text:
            text = text.split(sep, 1)[0]
            break
    lowered = text.lower()
    for sep in (" and ", " & "):
        if sep in lowered:
            text = text.split(sep, 1)[0]
            break
    return text.strip().title()


def _extract_mm_value(value: Any) -> Optional[float]:
    text = _safe_str(value).lower()
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*mm\b", text)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None


def _bucket_case_size(mm_value: Optional[float]) -> str:
    if mm_value is None:
        return ""
    if mm_value < 34:
        return "<34mm"
    if mm_value < 38:
        return "34-38mm"
    if mm_value < 42:
        return "38-42mm"
    if mm_value <= 46:
        return "42-46mm"
    return ">46mm"


def _normalize_movement_category(value: Any) -> str:
    text = _safe_str(value).lower()
    if not text:
        return ""
    if any(token in text for token in ("hybrid", "solar", "kinetic", "eco-drive", "spring drive")):
        return "Hybrid"
    if any(token in text for token in ("automatic", "self-winding", "self winding")):
        return "Automatic"
    if any(token in text for token in ("manual", "hand-wound", "hand wound")):
        return "Manual"
    if "quartz" in text:
        return "Quartz"
    return ""


def _normalize_feature_tokens(values: Sequence[str]) -> List[str]:
    keywords = (
        "chronograph",
        "gmt",
        "moonphase",
        "tourbillon",
        "date",
        "day-date",
        "power reserve",
        "perpetual calendar",
        "annual calendar",
        "world time",
        "alarm",
        "tachymeter",
        "chronometer",
        "skeleton",
        "open heart",
        "diver",
        "rotating bezel",
        "flyback",
        "split seconds",
        "minute repeater",
    )
    tokens: List[str] = []
    for item in _clean_list(values):
        text = item.lower()
        item_tokens: List[str] = []
        for match in re.finditer(r"\b\d+\s*(atm|bar|m)\b", text):
            item_tokens.append(match.group(0).replace(" ", ""))
        for keyword in keywords:
            if keyword in text:
                item_tokens.append(keyword)
        if not item_tokens:
            cleaned = re.sub(r"[^a-z0-9]+", " ", text).strip()
            if cleaned and len(cleaned.split()) <= 3:
                item_tokens.append(cleaned)
        tokens.extend(item_tokens)
    return _dedupe_preserve(tokens)


def _connect_option_value(target_url: Any) -> Dict[str, str]:
    url = _safe_str(target_url)
    if url and url.startswith("https://"):
        safe_url = url.replace('"', "&quot;")
        return {"Option1 Name": "BRAND URL", "Option1 Value": f'<a href="{safe_url}">CONNECT</a>'}
    return {"Option1 Name": "Title", "Option1 Value": "Default Title"}


def _align_weight_units(row: Dict[str, str]) -> None:
    # Keep Variant Grams as grams without converting to kilograms.
    raw = row.get("Variant Grams", "").strip()
    if not raw:
        row["Variant Weight Unit"] = row.get("Variant Weight Unit", "")
        return
    try:
        grams_value = float(re.sub(r"[^0-9.]+", "", raw) or 0.0)
    except ValueError:
        row["Variant Grams"] = ""
        row["Variant Weight Unit"] = ""
        return
    if grams_value <= 0:
        row["Variant Grams"] = ""
        row["Variant Weight Unit"] = ""
        return
    row["Variant Grams"] = f"{grams_value:g}"
    row["Variant Weight Unit"] = row.get("Variant Weight Unit", "")


def _prepare_options(value: Any) -> List[Dict[str, str]]:
    # Normalise variants into a list of {name, value} pairs.
    if value is None:
        return []
    normalized: List[Dict[str, str]] = []
    if isinstance(value, dict):
        candidate = {
            "name": _safe_str(value.get("name")),
            "value": _safe_str(value.get("value")),
        }
        if candidate["name"] or candidate["value"]:
            normalized.append(candidate)
        return normalized
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                candidate = {
                    "name": _safe_str(item.get("name")),
                    "value": _safe_str(item.get("value")),
                }
                if candidate["name"] or candidate["value"]:
                    normalized.append(candidate)
            else:
                text = _safe_str(item)
                if text:
                    normalized.append({"name": "Option", "value": text})
    elif isinstance(value, str):
        text = _safe_str(value)
        if text:
            normalized.append({"name": "Option", "value": text})
    return normalized[:3]


def map_raw_to_shopify(headers: Sequence[str], raw: Dict[str, Any], url: str) -> Dict[str, str]:
    # Start with an empty row that already carries Shopify-safe defaults.
    normalised = {header: "" for header in headers}
    normalised.update({k: str(v) for k, v in SHOPIFY_DEFAULTS.items() if k in normalised})

    # Use the provided handle or fall back to a slug made from the URL.
    handle = _slugify(raw.get("handle")) or _slugify(_slug_from_url(url))
    normalised["Handle"] = handle

    # Copy over the product title and reuse it for SEO when possible.
    title = _safe_str(raw.get("title"))
    if title:
        normalised["Title"] = title
        if "SEO Title" in normalised:
            normalised["SEO Title"] = title

    # Prefer HTML descriptions, but wrap plain text in a simple paragraph.
    body_html = raw.get("body_html") or ""
    body_html = body_html if isinstance(body_html, str) else ""
    if not body_html:
        plain = _safe_str(raw.get("plain_description"))
        if plain:
            body_html = f"<p>{plain}</p>"
    if body_html:
        normalised["Body (HTML)"] = body_html

    # Bring over vendor and category wording straight from the site.
    vendor = _safe_str(raw.get("vendor"))
    if vendor:
        normalised["Vendor"] = vendor

    normalised["Product Category"] = REQUIRED_PRODUCT_CATEGORY

    # Use the brand's product type when it exists, otherwise fall back to vendor.
    product_type = _safe_str(raw.get("product_type"))
    if product_type:
        normalised["Type"] = product_type
    elif vendor:
        normalised["Type"] = vendor
    # Align with Shopify export constraints on owner subtype.
    if "Type" in normalised:
        normalised["Type"] = REQUIRED_PRODUCT_TYPE

    # Tags hold loose keywords plus the detected currency for easy filtering.
    tags = _dedupe_preserve(_clean_list(raw.get("tags")))
    currency_code = _safe_str(raw.get("currency")).upper()
    if currency_code and re.fullmatch(r"[A-Z]{3}", currency_code) and currency_code not in tags:
        tags.append(currency_code)
    if tags:
        normalised["Tags"] = ", ".join(tags)

    # Copy the reference number into both SKU and MPN style fields.
    reference = _safe_str(raw.get("reference_number"))
    if reference:
        normalised["Variant SKU"] = reference
        if "Google Shopping / MPN" in normalised:
            normalised["Google Shopping / MPN"] = reference

    # Extract numbers for price fields so Shopify can parse them safely.
    price = _normalise_numeric(raw.get("price"))
    if price:
        normalised["Variant Price"] = price

    compare_price = _normalise_numeric(raw.get("compare_at_price"))
    if compare_price:
        normalised["Variant Compare At Price"] = compare_price

    # Respect explicit shipping and tax flags when the site states them.
    shipping = _coerce_bool(raw.get("requires_shipping"))
    if shipping:
        normalised["Variant Requires Shipping"] = shipping

    taxable = _coerce_bool(raw.get("taxable"))
    if taxable:
        normalised["Variant Taxable"] = taxable

    # Pull across identifiers that help with inventory tracking.
    barcode = _safe_str(raw.get("barcode"))
    if barcode:
        normalised["Variant Barcode"] = barcode

    grams = _normalise_numeric(raw.get("grams"))
    if grams:
        normalised["Variant Grams"] = grams

    # Use a short SEO description, falling back to the plain copy if needed.
    seo_description = _safe_str(raw.get("seo_description"))
    if not seo_description:
        plain = _safe_str(raw.get("plain_description"))
        if plain:
            seo_description = plain[:320]
    if seo_description:
        normalised["SEO Description"] = seo_description

    # Collection and material details feed custom metafields.
    collection = _safe_str(raw.get("collection"))
    if collection:
        normalised["Collection (product.metafields.custom.collection)"] = collection

    case_material_raw = _safe_str(raw.get("case_material"))
    if case_material_raw:
        case_material, case_coating = _split_case_material(case_material_raw)
        if case_material:
            normalised["Case Material (product.metafields.custom.case_material)"] = case_material
        else:
            normalised["Case Material (product.metafields.custom.case_material)"] = case_material_raw
        if case_coating:
            normalised[build_metafield_header("Case Coating", "case_coating")] = case_coating

    # Movement details can live under either key, so check both.
    movement = _normalize_movement_category(raw.get("movement") or raw.get("caliber_type"))
    if movement:
        normalised["Movement (product.metafields.custom.movement)"] = movement

    strap_type_raw = _safe_str(raw.get("strap_type"))
    if strap_type_raw:
        strap_clean, strap_notes = _split_strap_value(strap_type_raw)
        normalised["Strap (product.metafields.custom.strap)"] = strap_clean or strap_type_raw
        if strap_notes:
            normalised[build_metafield_header("Strap Notes", "strap_notes")] = strap_notes

    # Strap and dial colours feed multiple metafields, so reuse the value.
    strap_color = _simplify_color(raw.get("strap_color"))
    dial_color = _simplify_color(raw.get("dial_color"))
    if strap_color:
        if "Strap Color (product.metafields.custom.strap_color)" in normalised:
            normalised["Strap Color (product.metafields.custom.strap_color)"] = strap_color

    # Capture water resistance from whichever field the site exposes.
    water_resistance = _safe_str(raw.get("water_resistance")) or _safe_str(raw.get("water_resistance_m"))
    if water_resistance:
        normalised["Water Resistance (product.metafields.custom.water_resistance)"] = water_resistance

    watch_display = _safe_str(raw.get("watch_display"))
    if watch_display:
        normalised["Watch display (product.metafields.custom.watch_display)"] = watch_display

    target_gender = _normalize_target_gender(raw.get("gender"))
    if target_gender and "Target gender (product.metafields.shopify.target-gender)" in normalised:
        normalised["Target gender (product.metafields.shopify.target-gender)"] = target_gender

    age_group = _normalize_age_group(raw.get("age_group"))
    if age_group and "Age group (product.metafields.shopify.age-group)" in normalised:
        normalised["Age group (product.metafields.shopify.age-group)"] = age_group

    # Style and condition only copy over when spelled out on the site.

    style = _safe_str(raw.get("style"))
    if style:
        normalised["Style (product.metafields.custom.style)"] = style

    condition = _safe_str(raw.get("condition"))
    if condition and "Condition (product.metafields.custom.condition)" in normalised:
        normalised["Condition (product.metafields.custom.condition)"] = condition

    edition_raw = _safe_str(raw.get("edition_type"))
    if edition_raw:
        edition_type, edition_number = _split_edition_type(edition_raw)
        normalised["Edition Type (product.metafields.custom.edition_type)"] = edition_type or edition_raw
        if edition_number:
            normalised[build_metafield_header("Edition Number", "edition_number")] = edition_number

    # Merge different feature lists so we only mention each perk once.
    features = _clean_list(raw.get("other_features"))
    complications = _clean_list(raw.get("complications"))
    combined_features = _normalize_feature_tokens(features + complications)
    if combined_features:
        normalised["Watch features (product.metafields.custom.watch_features)"] = "; ".join(combined_features)

    target_url = (
        _safe_str(raw.get("official_site_url"))
        or _safe_str(raw.get("official_url"))
        or _safe_str(raw.get("brand_url"))
        or _safe_str(raw.get("source_url"))
        or url
    )
    normalised.update(_connect_option_value(target_url))
    for key in (
        "Option2 Name",
        "Option2 Value",
        "Option2 Linked To",
        "Option3 Name",
        "Option3 Value",
        "Option3 Linked To",
    ):
        if key in normalised:
            normalised[key] = ""

    # Finish by standardising weights.
    _align_weight_units(normalised)

    # Guarantee Shopify sees the usual shipping/tax defaults when nothing was stated.
    if not normalised.get("Variant Requires Shipping"):
        normalised["Variant Requires Shipping"] = "true"
    if not normalised.get("Variant Taxable"):
        normalised["Variant Taxable"] = "true"
    return normalised


def _prepare_variant_records(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalise the raw['variants'] list into a safe sequence of variant descriptors.

    When the model does not emit a dedicated variants list, fall back to a single
    synthetic variant built from the product-level fields so behaviour matches the
    original single-variant mapping.
    """
    variants_value = raw.get("variants")
    records: List[Dict[str, Any]] = []
    if isinstance(variants_value, list):
        for item in variants_value:
            if isinstance(item, dict):
                records.append(item)
            if len(records) >= MAX_VARIANTS:
                break
    if records:
        return records
    # Fallback: treat the whole product as a single implicit variant.
    return [
        {
            "sku": raw.get("reference_number"),
            "price": raw.get("price"),
            "compare_at_price": raw.get("compare_at_price"),
            "grams": raw.get("grams"),
            "requires_shipping": raw.get("requires_shipping"),
            "taxable": raw.get("taxable"),
            "barcode": raw.get("barcode"),
            "options": raw.get("options"),
            "images": raw.get("images"),
            "image_alt_text": raw.get("image_alt_text"),
        }
    ]


def _extract_variant_handle_parts(variant: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, str]:
    color = ""
    strap = ""
    options = _prepare_options(variant.get("options"))
    for option in options:
        name = option.get("name", "").lower()
        value = option.get("value", "")
        if not value:
            continue
        if not color and ("color" in name or "dial" in name):
            color = value
        if not strap and any(token in name for token in ("strap", "band", "bracelet")):
            strap = value
    if not color:
        color = _safe_str(raw.get("dial_color")) or _safe_str(raw.get("strap_color"))
    if not strap:
        strap = _safe_str(raw.get("strap_type")) or _safe_str(raw.get("strap_color"))
    return {"color": color, "strap": strap}


def _dedupe_handle(handle: str, seen: set[str]) -> str:
    if handle not in seen:
        return handle
    counter = 2
    while True:
        candidate = f"{handle}-{counter}"
        if candidate not in seen:
            return candidate
        counter += 1


def validate_row(row: Dict[str, str]) -> bool:
    if row.get("Product Category") != REQUIRED_PRODUCT_CATEGORY:
        return False
    if not row.get("Handle"):
        return False
    if row.get("Option1 Name") == "BRAND URL":
        value = (row.get("Option1 Value") or "").strip()
        if not (value.startswith('<a href="https://') and value.endswith(">CONNECT</a>")):
            return False
        lowered = value.lower()
        if "<script>" in lowered or "javascript:" in lowered or "onclick" in lowered:
            return False

    gender_key = "Target gender (product.metafields.shopify.target-gender)"
    if gender_key in row and row.get(gender_key) not in {"Men", "Women", "Unisex"}:
        row[gender_key] = ""

    grams_raw = (row.get("Variant Grams") or "").strip()
    if grams_raw:
        try:
            grams_value = float(re.sub(r"[^0-9.]+", "", grams_raw) or 0.0)
        except ValueError:
            grams_value = 0.0
        if grams_value < 1:
            grams_value *= 1000.0
        if grams_value > 0:
            row["Variant Grams"] = str(int(round(grams_value)))
        else:
            row["Variant Grams"] = ""
    return True


def _extract_canonical_features_from_raw(raw: Dict[str, Any]) -> tuple[Dict[str, str], List[Dict[str, str]]]:
    """Derive canonical watch features from multiple raw signals.

    We merge:
    - The model-emitted `feature_pairs` list (explicit label/value specs).
    - The dedicated raw fields (case_material, water_resistance, etc.) for resilience.
    """
    canonical, unmapped = canonicalize_watch_feature_pairs(raw.get("feature_pairs"))

    def _merge(key: str, value: Any) -> None:
        text = _safe_str(value)
        if not text:
            return
        canonical[key] = _merge_feature_values(canonical.get(key, ""), text)

    _merge("dial_color", raw.get("dial_color"))
    _merge("strap_color", raw.get("strap_color"))
    _merge("movement", raw.get("movement"))
    _merge("caliber", raw.get("caliber_type"))
    _merge("water_resistance", raw.get("water_resistance") or raw.get("water_resistance_m"))
    _merge("watch_display", raw.get("watch_display"))

    # Dimensions: avoid merging messy multi-dimension strings into a single field.
    case_size_raw = _safe_str(raw.get("case_size"))
    watch_case_diameter_raw = _safe_str(raw.get("watch_case_diameter"))
    case_thickness_raw = _safe_str(raw.get("case_thickness"))
    case_length_raw = _safe_str(raw.get("case_length"))
    # Prefer explicit fields first.
    if watch_case_diameter_raw and not canonical.get("case_diameter"):
        canonical["case_diameter"] = watch_case_diameter_raw
    if case_thickness_raw and not canonical.get("case_thickness"):
        canonical["case_thickness"] = case_thickness_raw
    if case_length_raw and not canonical.get("lug_to_lug"):
        canonical["lug_to_lug"] = case_length_raw
    # Fallback: parse "40 x 42 x 11mm" style blocks.
    if case_size_raw:
        d_mm, l2l_mm, thick_mm = _extract_case_dimensions(case_size_raw)
        if d_mm and not canonical.get("case_diameter"):
            canonical["case_diameter"] = d_mm
        if l2l_mm and not canonical.get("lug_to_lug"):
            canonical["lug_to_lug"] = l2l_mm
        if thick_mm and not canonical.get("case_thickness"):
            canonical["case_thickness"] = thick_mm

    complications = _clean_list(raw.get("complications"))
    if complications:
        filtered = [item for item in complications if not _is_non_feature_text(item)]
        if filtered:
            _merge("complications", "; ".join(filtered))

    # Normalise key dimension fields to a single mm token where possible.
    if canonical.get("case_diameter"):
        parsed = _extract_case_dimensions(canonical["case_diameter"])[0]
        canonical["case_diameter"] = parsed or _normalise_dimension_mm(canonical["case_diameter"], (28.0, 52.0)) or canonical["case_diameter"]
    if canonical.get("case_thickness"):
        canonical["case_thickness"] = _normalise_dimension_mm(canonical["case_thickness"], (5.0, 25.0)) or canonical["case_thickness"]
    if canonical.get("lug_to_lug"):
        canonical["lug_to_lug"] = _normalise_dimension_mm(canonical["lug_to_lug"], (35.0, 65.0)) or canonical["lug_to_lug"]

    # Only keep keys that exist in the canonical schema so column naming stays stable.
    canonical = {k: v for k, v in canonical.items() if k in CANONICAL_WATCH_FEATURE_BY_KEY and v.strip()}
    return canonical, unmapped


def _apply_canonical_features_to_row(
    row: Dict[str, str],
    canonical_features: Dict[str, str],
) -> None:
    for key, value in canonical_features.items():
        feature = CANONICAL_WATCH_FEATURE_BY_KEY.get(key)
        if not feature:
            continue
        header = build_metafield_header(feature.label, feature.key, namespace=feature.namespace)
        existing = (row.get(header) or "").strip()
        incoming = (value or "").strip()
        if existing:
            # Never overwrite an existing value coming from the base mapping.
            continue
        row[header] = incoming


def map_raw_to_shopify_rows(headers: Sequence[str], raw: Dict[str, Any], url: str) -> List[Dict[str, str]]:
    """Expand a raw product payload into one or more Shopify-ready CSV rows.

    Multi-variant products become multiple standalone products (unique handles),
    each with its own image set. Single-variant products produce a single row.
    """
    variants = _prepare_variant_records(raw)
    rows: List[Dict[str, str]] = []
    base_handle = _slugify(raw.get("handle")) or _slugify(_slug_from_url(url))
    base_images = _clean_image_list(raw.get("images"))
    seen_handles: set[str] = set()
    canonical_features, _unmapped_pairs = _extract_canonical_features_from_raw(raw)

    for index, variant in enumerate(variants, start=1):
        # Merge variant-specific details onto the product-level raw payload.
        merged: Dict[str, Any] = dict(raw)
        if isinstance(variant, dict):
            sku = variant.get("sku")
            if sku:
                merged["reference_number"] = sku
            for key in (
                "price",
                "compare_at_price",
                "grams",
                "requires_shipping",
                "taxable",
                "barcode",
                "options",
                "image_alt_text",
            ):
                if key in variant and variant[key] not in (None, ""):
                    merged[key] = variant[key]

        handle_parts = _extract_variant_handle_parts(variant, merged)
        color_slug = _slugify(_simplify_color(handle_parts["color"]))
        strap_slug = _slugify(handle_parts["strap"])
        handle = "-".join(part for part in [base_handle, color_slug, strap_slug] if part)
        fallback_key = merged.get("reference_number") or merged.get("sku") or str(index)
        if handle == base_handle and fallback_key:
            fallback_slug = _slugify(fallback_key)
            if handle and fallback_slug:
                handle = f"{handle}-{fallback_slug}"
            elif fallback_slug:
                handle = fallback_slug
        handle = _dedupe_handle(handle, seen_handles)
        seen_handles.add(handle)
        merged["handle"] = handle

        row = map_raw_to_shopify(headers, merged, url)
        _apply_canonical_features_to_row(row, canonical_features)
        _enrich_watch_features(row, canonical_features)
        if not (row.get("Variant Grams") or "").strip():
            grams_from_weight = _extract_grams_from_text(canonical_features.get("weight", ""))
            if grams_from_weight:
                row["Variant Grams"] = grams_from_weight

        variant_images = _clean_image_list(variant.get("images")) if isinstance(variant, dict) else []
        all_images = _dedupe_preserve(variant_images + base_images)
        hero_image = variant_images[0] if variant_images else (all_images[0] if all_images else "")
        image_alt = _safe_str(merged.get("image_alt_text")) or row.get("Title") or handle

        if hero_image:
            row["Image Src"] = hero_image
            row["Image Position"] = "1"
            row["Variant Image"] = hero_image
            if image_alt:
                row["Image Alt Text"] = image_alt

        extra_rows: List[Dict[str, str]] = []
        image_position = 1
        for img in all_images:
            if img == hero_image:
                continue
            image_position += 1
            extra = {header: "" for header in headers}
            extra["Handle"] = handle
            extra["Product Category"] = REQUIRED_PRODUCT_CATEGORY
            if "Type" in extra:
                extra["Type"] = REQUIRED_PRODUCT_TYPE
            extra["Option1 Name"] = ""
            extra["Option1 Value"] = ""
            extra["Image Src"] = img
            extra["Image Position"] = str(image_position)
            if image_alt:
                extra["Image Alt Text"] = image_alt
            extra_rows.append(extra)

        if validate_row(row):
            rows.append(row)
            for extra in extra_rows:
                if validate_row(extra):
                    rows.append(extra)
    return rows


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def process_product(
    session: requests.Session,
    model: genai.GenerativeModel,
    headers: Sequence[str],
    url: str,
    browser_context: Any = None,
) -> List[Dict[str, str]]:
    print(f"[product] {url}")
    # Fetch the page, clean it up, and ask Gemini for structured data.
    html = fetch_html(session, url, browser_context=browser_context)
    payload = _clean_html(html, url)
    payload.url = url
    prompt = build_prompt(headers, payload)
    raw_row = invoke_model(model, prompt)

    # Skip non-product pages (brand home, category listings, etc.) using the model's own signal.
    is_product = raw_row.get("is_product_page", True)
    if isinstance(is_product, str):
        is_product = is_product.strip().lower() in {"true", "1", "yes"}
    heuristic_product = _looks_like_single_product_page(html)
    if not (is_product or heuristic_product):
        raise ValueError("Page is not a single watch product detail page")

    # Double-check the currency in case the model left it blank.
    detected_currency = _safe_str(raw_row.get("currency")).upper()
    if not detected_currency:
        detected_currency = _extract_currency_from_html(html)
    if detected_currency:
        raw_row["currency"] = detected_currency

    # Weight is often present in spec tables; fill it deterministically when the model misses it.
    if not _safe_str(raw_row.get("grams")):
        grams = _extract_grams_from_html(html)
        if grams:
            raw_row["grams"] = grams

    return map_raw_to_shopify_rows(headers, raw_row, url)


def write_csv(rows: Iterable[Dict[str, str]], headers: Sequence[str], path: Path) -> None:
    # Write the Shopify-ready rows to disk using the exact header order.
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, lineterminator="\r\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def _is_metafield_header(header: str) -> bool:
    return "(product.metafields." in header


_METAFIELD_KEY_RE = re.compile(r"\(product\.metafields\.[^.]+\.(?P<key>[^)]+)\)\s*$")
_CANONICAL_FEATURE_ORDER: Dict[str, int] = {
    feature.key: index for index, feature in enumerate(CANONICAL_WATCH_FEATURES)
}


def _metafield_sort_key(header: str) -> tuple[int, str]:
    match = _METAFIELD_KEY_RE.search(header or "")
    key = match.group("key") if match else ""
    normalized_key = key.replace("-", "_").lower()
    base_order = _CANONICAL_FEATURE_ORDER.get(normalized_key)
    if base_order is not None:
        return (base_order * 100, (header or "").lower())

    suffix_offsets = (
        ("_details", 50),
        ("_notes", 60),
        ("_number", 70),
    )
    for suffix, offset in suffix_offsets:
        if normalized_key.endswith(suffix):
            base_key = normalized_key[: -len(suffix)]
            base_order = _CANONICAL_FEATURE_ORDER.get(base_key)
            if base_order is not None:
                return (base_order * 100 + offset, (header or "").lower())
            base_order = _CANONICAL_FEATURE_ORDER.get(f"{base_key}_type")
            if base_order is not None:
                return (base_order * 100 + offset, (header or "").lower())
            break

    return (10_000 * 100, (header or "").lower())


def build_brand_headers(base_headers: Sequence[str], rows: Sequence[Dict[str, str]]) -> List[str]:
    """Return headers with only the metafield columns that appear for this brand/run."""
    used_metafields: set[str] = set()
    for row in rows:
        for header, value in row.items():
            if not _is_metafield_header(header):
                continue
            if str(value).strip():
                used_metafields.add(header)

    if not used_metafields:
        return [h for h in base_headers if not _is_metafield_header(h)]

    meta_indices = [idx for idx, header in enumerate(base_headers) if _is_metafield_header(header)]
    if meta_indices:
        first = min(meta_indices)
        last = max(meta_indices)
        prefix = list(base_headers[:first])
        suffix = list(base_headers[last + 1 :])
        base_meta = [h for h in base_headers[first : last + 1] if _is_metafield_header(h)]
    else:
        prefix = list(base_headers)
        suffix = []
        base_meta = []

    base_meta_used = [h for h in base_meta if h in used_metafields]
    dynamic_meta = sorted((h for h in used_metafields if h not in set(base_meta)), key=_metafield_sort_key)
    return prefix + base_meta_used + dynamic_meta + suffix


def build_full_headers(base_headers: Sequence[str], rows: Sequence[Dict[str, str]]) -> List[str]:
    """Return headers that keep the full base schema but still include dynamic metafields."""
    used_metafields: set[str] = set()
    for row in rows:
        for header, value in row.items():
            if not _is_metafield_header(header):
                continue
            if str(value).strip():
                used_metafields.add(header)

    meta_indices = [idx for idx, header in enumerate(base_headers) if _is_metafield_header(header)]
    if meta_indices:
        first = min(meta_indices)
        last = max(meta_indices)
        prefix = list(base_headers[:first])
        suffix = list(base_headers[last + 1 :])
        base_meta = [h for h in base_headers[first : last + 1] if _is_metafield_header(h)]
    else:
        prefix = list(base_headers)
        suffix = []
        base_meta = []

    dynamic_meta = sorted((h for h in used_metafields if h not in set(base_meta)), key=_metafield_sort_key)
    return prefix + base_meta + dynamic_meta + suffix


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    # Command-line arguments let the user pick URLs, output path, and pacing.
    parser = argparse.ArgumentParser(description="AI-powered watch scraper")
    parser.add_argument("--catalog-url", help="Catalog/collection URL to discover product links")
    parser.add_argument(
        "--product-url",
        action="append",
        help="Specific product URL to scrape (can be repeated)",
    )
    parser.add_argument(
        "--max-products",
        type=int,
        default=3,
        help="Maximum number of products to scrape (0 = no limit)",
    )
    parser.add_argument(
        "--output",
        default="ai_watch_scraper.csv",
        help="Destination CSV path",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay (seconds) between product requests",
    )
    parser.add_argument(
        "--header-mode",
        choices=("brand", "full"),
        default="brand",
        help=(
            "CSV header strategy: 'brand' drops metafield columns that are completely empty for the current run "
            "(recommended for per-brand imports). 'full' keeps the entire base Shopify schema and appends any "
            "dynamic metafields that appear."
        ),
    )
    parser.add_argument(
        "--discover-from-product",
        action="store_true",
        help=(
            "When provided alongside --product-url, treat the first product URL as a seed, "
            "auto-discover a related catalog/collection page, and crawl all product links from it."
        ),
    )
    return parser.parse_args(argv)


def resolve_product_urls(
    session: requests.Session,
    catalog_url: Optional[str],
    explicit_urls: Optional[List[str]],
    max_products: int,
    browser_context: Any = None,
    discover_from_product: bool = False,
) -> List[str]:
    limit = max(0, max_products or 0)
    urls: List[str] = []

    def _limit_reached() -> bool:
        return limit > 0 and len(urls) >= limit

    # Optionally auto-discover a catalog page starting from a seed product URL.
    if discover_from_product and explicit_urls and not catalog_url:
        seed = explicit_urls[0]
        discovered = discover_catalog_from_product(
            session=session,
            product_url=seed,
            max_products=limit,
            browser_context=browser_context,
        )
        for url in discovered:
            if url not in urls:
                urls.append(url)
                if _limit_reached():
                    break
    else:
        # Start with any explicit product URLs the user typed in.
        if explicit_urls:
            urls.extend(explicit_urls)
            if _limit_reached():
                return urls[:limit]
        # If a catalog is supplied, crawl it until we hit the remaining quota.
        if catalog_url:
            discovery_limit = 0 if limit == 0 else max(0, limit - len(urls))
            discovered = discover_product_links(
                session, catalog_url, limit=discovery_limit, browser_context=browser_context
            )
            for url in discovered:
                if url not in urls:
                    urls.append(url)
                    if _limit_reached():
                        break
    if limit > 0:
        return urls[:limit]
    return urls


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    base_headers = list(SHOPIFY_HEADERS)
    session = _build_session()
    model = configure_model()
    with _playwright_context() as browser_context:
        # Figure out which product pages to scrape based on user input.
        product_urls = resolve_product_urls(
            session=session,
            catalog_url=args.catalog_url,
            explicit_urls=args.product_url,
            max_products=args.max_products,
            browser_context=browser_context,
            discover_from_product=args.discover_from_product,
        )

        if not product_urls:
            raise SystemExit("No product URLs provided or discovered.")

        rows: List[Dict[str, str]] = []
        # Loop through each product URL with a gentle delay to stay polite.
        for url in product_urls:
            try:
                product_rows = process_product(
                    session,
                    model,
                    base_headers,
                    url,
                    browser_context=browser_context,
                )
                rows.extend(product_rows)
            except Exception as exc:  # pragma: no cover - runtime safety
                print(f"  ! Failed to process {url}: {exc}")
            time.sleep(max(0.0, args.delay) + random.uniform(0.3, 0.8))

        if not rows:
            raise SystemExit("No rows were generated.")

        # Save the finished CSV and let the user know where it landed.
        output_path = Path(args.output)
        output_headers = (
            build_brand_headers(base_headers, rows)
            if args.header_mode == "brand"
            else build_full_headers(base_headers, rows)
        )
        write_csv(rows, output_headers, output_path)
        print(f"Wrote {len(rows)} rows → {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Aborted by user")

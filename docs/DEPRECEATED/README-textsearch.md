# Product Image Search Script

Builds a web search query from product attributes and opens image search results in the default browser.

## Usage

```bash
python text_based_search.py [OPTIONS]
```

At least one argument is required.

## Options

| Flag | Short | Description |
|------|-------|-------------|
| `--item-name` | `-n` | Product / item name |
| `--item-size` | `-s` | Item size (e.g. `10mm`, `Large`) |
| `--manufacture-number` | `-mn` | Manufacturer part / model number |
| `--manufacture-vin` | `-mv` | Manufacturer VIN / serial |
| `--enterprise-product-number` | `-ep` | Enterprise product number |
| `--enterprise-vin` | `-ev` | Enterprise VIN |
| `--product-description` | `-d` | Product description (first 8 words used) |
| `--engine` | `-e` | `google`, `bing`, `duckduckgo`, or `all` (default: `google`) |
| `--print-only` | `-p` | Print URLs instead of opening in browser |

## Examples

```bash
# Search by name
python text_based_search.py -n "stainless steel bolt"

# Search by manufacturer number and size
python text_based_search.py -mn "AB-1234" -s "10mm"

# Print URLs for all engines without opening browser
python text_based_search.py -n "widget" -e all -p
```

## Requirements

- Python 3.x (no external dependencies)

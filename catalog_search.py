#!/usr/bin/env python3
"""
Catalog Database Search Tool
Searches through an Excel catalog database and matches entries with corresponding images.
"""

import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Tuple
import re


class CatalogSearch:
    def __init__(self, catalog_path: str, image_directory: str = None):
        """
        Initialize the catalog search tool.
        
        Args:
            catalog_path: Path to Excel file containing the catalog
            image_directory: Directory containing product images (optional)
        """
        self.catalog_path = catalog_path
        self.image_directory = image_directory
        self.df = None
        self.load_catalog()
        
    def load_catalog(self):
        """Load the catalog from Excel file."""
        try:
            self.df = pd.read_excel(self.catalog_path)
            print(f"✓ Loaded catalog with {len(self.df)} entries")
            print(f"✓ Columns: {', '.join(self.df.columns.tolist())}")
        except Exception as e:
            print(f"Error loading catalog: {e}")
            raise
    
    def search(self, query: str, columns: List[str] = None, 
               case_sensitive: bool = False, exact_match: bool = False) -> pd.DataFrame:
        """
        Search the catalog for matching entries.
        
        Args:
            query: Search term
            columns: Specific columns to search (if None, searches all text columns)
            case_sensitive: Whether to perform case-sensitive search
            exact_match: Whether to require exact match or partial match
            
        Returns:
            DataFrame with matching entries
        """
        if self.df is None:
            print("Catalog not loaded!")
            return pd.DataFrame()
        
        # If no columns specified, search all object (text) columns
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Validate columns exist
        valid_columns = [col for col in columns if col in self.df.columns]
        if not valid_columns:
            print(f"Warning: No valid columns found from {columns}")
            return pd.DataFrame()
        
        # Create search mask
        mask = pd.Series([False] * len(self.df))
        
        for col in valid_columns:
            col_data = self.df[col].astype(str)
            
            if not case_sensitive:
                col_data = col_data.str.lower()
                search_term = query.lower()
            else:
                search_term = query
            
            if exact_match:
                mask |= (col_data == search_term)
            else:
                mask |= col_data.str.contains(search_term, na=False, regex=False)
        
        results = self.df[mask]
        return results
    
    def search_multi_criteria(self, criteria: Dict[str, str]) -> pd.DataFrame:
        """
        Search using multiple criteria (AND logic).
        
        Args:
            criteria: Dictionary mapping column names to search values
            
        Returns:
            DataFrame with matching entries
        """
        if self.df is None:
            return pd.DataFrame()
        
        mask = pd.Series([True] * len(self.df))
        
        for column, value in criteria.items():
            if column in self.df.columns:
                col_data = self.df[column].astype(str).str.lower()
                search_term = str(value).lower()
                mask &= col_data.str.contains(search_term, na=False, regex=False)
            else:
                print(f"Warning: Column '{column}' not found in catalog")
        
        return self.df[mask]
    
    def find_images(self, results: pd.DataFrame, 
                   id_column: str = None, 
                   image_extensions: List[str] = None) -> Dict[int, List[str]]:
        """
        Find images corresponding to search results.
        
        Args:
            results: DataFrame of search results
            id_column: Column name containing product/tool ID or name
            image_extensions: List of image file extensions to search for
            
        Returns:
            Dictionary mapping result index to list of image paths
        """
        if self.image_directory is None:
            print("No image directory specified")
            return {}
        
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        
        # Auto-detect ID column if not specified
        if id_column is None:
            # Look for common ID column names
            possible_cols = ['id', 'product_id', 'tool_id', 'sku', 'name', 'product_name']
            for col in possible_cols:
                if col.lower() in [c.lower() for c in results.columns]:
                    id_column = col
                    break
            
            if id_column is None and len(results.columns) > 0:
                id_column = results.columns[0]
                print(f"Using '{id_column}' as ID column")
        
        if id_column not in results.columns:
            print(f"Column '{id_column}' not found in results")
            return {}
        
        image_map = {}
        image_dir = Path(self.image_directory)
        
        if not image_dir.exists():
            print(f"Image directory not found: {self.image_directory}")
            return {}
        
        # Get all image files in directory
        all_images = []
        for ext in image_extensions:
            all_images.extend(image_dir.glob(f"**/*{ext}"))
            all_images.extend(image_dir.glob(f"**/*{ext.upper()}"))
        
        # Match images to results
        for idx, row in results.iterrows():
            identifier = str(row[id_column])
            matched_images = []
            
            for img_path in all_images:
                # Check if identifier appears in image filename
                if identifier.lower() in img_path.stem.lower():
                    matched_images.append(str(img_path))
            
            if matched_images:
                image_map[idx] = matched_images
        
        return image_map
    
    def display_results(self, results: pd.DataFrame, 
                       show_images: bool = True,
                       id_column: str = None,
                       max_results: int = 10):
        """
        Display search results in a formatted way.
        
        Args:
            results: DataFrame of search results
            show_images: Whether to find and display associated images
            id_column: Column to use for image matching
            max_results: Maximum number of results to display
        """
        if results.empty:
            print("\n❌ No matches found")
            return
        
        print(f"\n✓ Found {len(results)} matching entries")
        print("=" * 80)
        
        display_count = min(len(results), max_results)
        
        for i, (idx, row) in enumerate(results.head(display_count).iterrows(), 1):
            print(f"\n[{i}] Entry #{idx}")
            print("-" * 80)
            
            for col in results.columns:
                value = row[col]
                if pd.notna(value):
                    print(f"  {col}: {value}")
            
            if show_images and self.image_directory:
                image_map = self.find_images(results.loc[[idx]], id_column)
                if idx in image_map:
                    print(f"\n  📷 Associated Images:")
                    for img_path in image_map[idx]:
                        print(f"     - {img_path}")
        
        if len(results) > max_results:
            print(f"\n... and {len(results) - max_results} more results")


def main():
    """Example usage of the catalog search tool."""
    print("=" * 80)
    print("CATALOG DATABASE SEARCH TOOL")
    print("=" * 80)
    
    # Example usage
    catalog_file = input("\nEnter path to Excel catalog file: ").strip()
    
    if not os.path.exists(catalog_file):
        print(f"Error: File not found: {catalog_file}")
        return
    
    image_dir = input("Enter path to image directory (press Enter to skip): ").strip()
    if not image_dir:
        image_dir = None
    
    # Initialize search tool
    try:
        searcher = CatalogSearch(catalog_file, image_dir)
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return
    
    # Interactive search loop
    while True:
        print("\n" + "=" * 80)
        print("SEARCH OPTIONS:")
        print("  1. Simple text search")
        print("  2. Multi-criteria search")
        print("  3. Show all entries")
        print("  4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            query = input("\nEnter search term: ").strip()
            if query:
                results = searcher.search(query)
                searcher.display_results(results)
        
        elif choice == '2':
            print("\nEnter search criteria (column=value), one per line.")
            print("Press Enter on empty line when done.")
            print(f"Available columns: {', '.join(searcher.df.columns.tolist())}\n")
            
            criteria = {}
            while True:
                entry = input("Criterion: ").strip()
                if not entry:
                    break
                
                if '=' in entry:
                    col, val = entry.split('=', 1)
                    criteria[col.strip()] = val.strip()
            
            if criteria:
                results = searcher.search_multi_criteria(criteria)
                searcher.display_results(results)
        
        elif choice == '3':
            searcher.display_results(searcher.df, max_results=20)
        
        elif choice == '4':
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()

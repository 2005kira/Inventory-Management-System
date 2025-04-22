import pytest
from app.app import load_data  # Assuming you move app.py to app/app.py

def test_data_loading():
    """Test that data loads correctly"""
    df = load_data()
    assert not df.empty, "Data should not be empty"
    assert 'CustomerID' in df.columns, "CustomerID column should exist"

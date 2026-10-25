import pytest
from integrations.utils import is_odd

def test_is_odd_positive_odd():
    assert is_odd(1) is True
    assert is_odd(3) is True
    assert is_odd(101) is True

def test_is_odd_positive_even():
    assert is_odd(2) is False
    assert is_odd(4) is False
    assert is_odd(100) is False

def test_is_odd_negative_odd():
    assert is_odd(-1) is True
    assert is_odd(-3) is True
    assert is_odd(-101) is True

def test_is_odd_negative_even():
    assert is_odd(-2) is False
    assert is_odd(-4) is False
    assert is_odd(-100) is False

def test_is_odd_zero():
    assert is_odd(0) is False
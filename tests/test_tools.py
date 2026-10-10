import pytest

from handbook_rag.tools import business_days_between, calculator, safe_eval


def test_calculator_basic():
    assert calculator.invoke({"expression": "3 * 250 + 4 * 70"}) == "1030"
    assert calculator.invoke({"expression": "25 - 10 / 4"}) == "22.5"
    assert calculator.invoke({"expression": "€2,000 + €1,500"}) == "3500"


@pytest.mark.parametrize("evil", ["__import__('os').system('echo hi')", "open('x')", "(1).__class__", "2 ** 9999", "a + 1"])
def test_calculator_rejects_unsafe_input(evil):
    with pytest.raises(Exception):
        safe_eval(evil)


def test_business_days_between():
    out = business_days_between.invoke({"start_date": "2026-07-06", "end_date": "2026-07-17"})
    assert out.startswith("10 working days")
    assert business_days_between.invoke({"start_date": "2026-07-04", "end_date": "2026-07-05"}).startswith("0 working")


def test_business_days_bad_range():
    with pytest.raises(ValueError):
        business_days_between.invoke({"start_date": "2026-07-10", "end_date": "2026-07-01"})

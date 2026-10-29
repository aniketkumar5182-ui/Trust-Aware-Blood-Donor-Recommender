from datetime import date, timedelta

import pytest

from app.compatibility import compatibility_score, compatible_donor_groups, is_compatible
from app.eligibility import check_eligibility

from .conftest import NOW

# donor -> set of recipients that can safely receive red cells
EXPECTED = {
    "O-": {"O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"},
    "O+": {"O+", "A+", "B+", "AB+"},
    "A-": {"A-", "A+", "AB-", "AB+"},
    "A+": {"A+", "AB+"},
    "B-": {"B-", "B+", "AB-", "AB+"},
    "B+": {"B+", "AB+"},
    "AB-": {"AB-", "AB+"},
    "AB+": {"AB+"},
}


@pytest.mark.parametrize("donor,recipients", EXPECTED.items())
def test_full_compatibility_matrix(donor, recipients):
    got = {r for r in EXPECTED if is_compatible(donor, r)}
    assert got == recipients


def test_compatible_donor_groups():
    assert set(compatible_donor_groups("AB-")) == {"O-", "A-", "B-", "AB-"}
    assert set(compatible_donor_groups("O-")) == {"O-"}
    assert len(compatible_donor_groups("AB+")) == 8


def test_unknown_group_rejected():
    with pytest.raises(ValueError):
        is_compatible("X+", "O+")


def test_compatibility_score_ordering():
    assert compatibility_score("A+", "A+") == 1.0
    assert compatibility_score("O+", "A+") == 0.75
    assert compatibility_score("O-", "A+") == 0.5      # scarce universal donor used as fallback
    assert compatibility_score("A+", "O+") == 0.0


def test_eligible_baseline(make_donor):
    assert check_eligibility(make_donor(), NOW).eligible


def test_recent_donation_blocks_by_sex(make_donor):
    d = (NOW - timedelta(days=100)).date()
    assert check_eligibility(make_donor(sex="M", last_donation_date=d), NOW).eligible          # 90-day rule
    res = check_eligibility(make_donor(sex="F", last_donation_date=d), NOW)                    # 120-day rule
    assert not res.eligible and res.days_until_eligible == 20


def test_margin_grows_after_interval(make_donor):
    near = check_eligibility(make_donor(last_donation_date=(NOW - timedelta(days=91)).date()), NOW)
    far = check_eligibility(make_donor(last_donation_date=(NOW - timedelta(days=200)).date()), NOW)
    never = check_eligibility(make_donor(last_donation_date=None), NOW)
    assert near.margin_score < far.margin_score <= never.margin_score == 1.0


@pytest.mark.parametrize("kw", [dict(age=17), dict(age=66), dict(weight_kg=44.0),
                                dict(is_active=False), dict(availability_status="unavailable")])
def test_hard_exclusions(make_donor, kw):
    res = check_eligibility(make_donor(**kw), NOW)
    assert not res.eligible and res.reasons

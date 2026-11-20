import pytest

pytest.importorskip("spacy")

from ariadne.verbatim_mapping.term_normalizer import TermNormalizer

SUBSTRINGS_TO_REMOVE = ["(disorder)", "(finding)", "unspecified", "initial encounter"]


@pytest.fixture(scope="module")
def normalizer():
    try:
        return TermNormalizer(SUBSTRINGS_TO_REMOVE)
    except IOError:
        pytest.skip("spaCy model 'en_core_web_sm' is not installed")


@pytest.mark.parametrize(
    "term, expected",
    [
        ("Liver-Disorders", "liver disorder"),
        ("Liver-Disorders (disorder)", "liver disorder"),
        ("Prinzmetal's angina", "prinzmetal angina"),
        ("Skin, disorder", "skin disorder"),
    ],
)
def test_normalize_term(normalizer, term, expected):
    assert normalizer.normalize_term(term) == expected


@pytest.mark.parametrize("term", ["Unspecified", "Initial encounter", "(disorder)"])
def test_term_consisting_only_of_removable_substrings_is_kept(normalizer, term):
    assert normalizer.normalize_term(term) != ""
    assert normalizer.normalize_terms([term], n_process=1)[0] != ""


def test_single_and_batch_normalization_agree(normalizer):
    terms = [
        "Liver-Disorders (disorder)",
        "Prinzmetal's angina",
        "Fractures of the left leg",
        "Severe depression (disorder)",
        "Unspecified",
        "Initial encounter",
        "Fever (finding)",
    ]
    batch = normalizer.normalize_terms(terms, n_process=1)
    single = [normalizer.normalize_term(term) for term in terms]
    assert batch == single

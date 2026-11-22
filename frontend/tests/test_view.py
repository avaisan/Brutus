from brutus_frontend.view import describe_counts, describe_person, plural
from tests.analysis_fixture import analysis


class TestPlural:
    def test_given_counts_when_pluralized_then_correct_forms(self) -> None:
        assert plural(1, "chair") == "1 chair"
        assert plural(2, "chair") == "2 chairs"
        assert plural(2, "person", "people") == "2 people"


class TestDescribeCounts:
    def test_given_people_animals_objects_when_described_then_summary(self) -> None:
        assert describe_counts(analysis()) == "2 people · 1 dog · 2 chairs"

    def test_given_empty_analysis_when_described_then_nothing_found(self) -> None:
        empty = analysis(
            people=[],
            animals=[],
            objects=[],
            people_count=0,
            animal_count=0,
            object_counts={},
        )
        assert describe_counts(empty) == "Nothing recognised"


class TestDescribePerson:
    def test_given_recognized_person_with_attributes_when_described_then_details(
        self,
    ) -> None:
        kirk = analysis()["people"][0]

        title, details = describe_person(kirk)

        assert title == "#1 · Kirk (98.7%)"
        assert "Age 30-40" in details
        assert "Happy (91.5%)" in details
        assert "Smiling" in details
        assert "Eyeglasses" in details
        assert "Beard" not in details

    def test_given_unknown_person_without_face_when_described_then_title_only(
        self,
    ) -> None:
        unknown = analysis()["people"][1]

        assert describe_person(unknown) == ("#2 · Unknown", ())

from dataclasses import dataclass
from typing import Tuple
import json
from enum import Enum

from fuzzysearch import find_near_matches
from rapidfuzz import fuzz


class TestType(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    ORDER = "order"


class ValidationError(Exception):
    """Exception raised for validation errors"""
    pass


@dataclass
class BasePDFTest:
    """Base class for all PDF test types"""
    pdf: str
    page: int
    id: str
    type: str
    threshold: float
    
    def __post_init__(self):
        # Validate common fields
        if not self.pdf:
            raise ValidationError("PDF filename cannot be empty")
        
        if not self.id:
            raise ValidationError("Test ID cannot be empty")
            
        if not isinstance(self.threshold, float) or not (0 <= self.threshold <= 1):
            raise ValidationError(f"Threshold must be a float between 0 and 1, got {self.threshold}")
        
        # Check that type is valid
        if self.type not in [t.value for t in TestType]:
            raise ValidationError(f"Invalid test type: {self.type}")
    
    def run(self, md_content: str) -> Tuple[bool, str]:
        """
        Run the test on the content of the provided .md file.
        Returns a tuple (passed, explanation) where 'passed' is True if the test passes,
        and 'explanation' is a short message explaining the failure when the test does not pass.
        """
        raise NotImplementedError("Subclasses must implement run method")


@dataclass
class TextPresenceTest(BasePDFTest):
    """Test for text presence or absence in a PDF"""
    text: str
    
    def __post_init__(self):
        super().__post_init__()
        
        # Additional validation for this specific test type
        if self.type not in [TestType.PRESENT.value, TestType.ABSENT.value]:
            raise ValidationError(f"Invalid type for TextPresenceTest: {self.type}")
            
        if not self.text.strip():
            raise ValidationError("Text field cannot be empty")
    
    def run(self, md_content: str) -> Tuple[bool, str]:
        reference_query = self.text
        threshold = self.threshold
        best_ratio = fuzz.partial_ratio(reference_query, md_content) / 100.0
        
        if self.type == TestType.PRESENT.value:
            if best_ratio >= threshold:
                return (True, "")
            else:
                return (False, f"Expected '{reference_query[:40]}...' with threshold {threshold} but best match ratio was {best_ratio:.3f}")
        else:  # absent
            if best_ratio < threshold:
                return (True, "")
            else:
                return (False, f"Expected absence of '{reference_query[:40]}...' with threshold {threshold} but best match ratio was {best_ratio:.3f}")


@dataclass
class TextOrderTest(BasePDFTest):
    """Test for text order in a PDF"""
    before: str
    after: str
    
    def __post_init__(self):
        super().__post_init__()
        
        # Additional validation for this specific test type
        if self.type != TestType.ORDER.value:
            raise ValidationError(f"Invalid type for TextOrderTest: {self.type}")
            
        if not self.before.strip():
            raise ValidationError("Before field cannot be empty")
            
        if not self.after.strip():
            raise ValidationError("After field cannot be empty")
    
    def run(self, md_content: str) -> Tuple[bool, str]:
        before = self.before
        after = self.after
        threshold = self.threshold
        max_l_dist = round((1.0 - threshold) * len(before))
        
        before_matches = find_near_matches(before, md_content, max_l_dist=max_l_dist)
        after_matches = find_near_matches(after, md_content, max_l_dist=max_l_dist)
        
        if not before_matches:
            return (False, f"'before' search text '{before[:40]}...' not found with max_l_dist {max_l_dist}")
        if not after_matches:
            return (False, f"'after' search text '{after[:40]}...' not found with max_l_dist {max_l_dist}")
        
        for before_match in before_matches:
            for after_match in after_matches:
                if before_match.start < after_match.start:
                    return (True, "")
        
        return (False, f"Could not find a location where '{before[:40]}...' appears before '{after[:40]}...'.")


def load_tests(jsonl_file: str) -> list[BasePDFTest]:
    """Load tests from a JSONL file"""
    tests = []
    
    with open(jsonl_file, 'r') as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                # Parse the JSON object
                data = json.loads(line)
                
                # Based on the type field, create the appropriate test object
                if data["type"] in [TestType.PRESENT.value, TestType.ABSENT.value]:
                    test = TextPresenceTest(
                        pdf=data["pdf"],
                        id=data["id"],
                        type=data["type"],
                        threshold=data["threshold"],
                        text=data["text"]
                    )
                elif data["type"] == TestType.ORDER.value:
                    test = TextOrderTest(
                        pdf=data["pdf"],
                        id=data["id"],
                        type=data["type"],
                        threshold=data["threshold"],
                        before=data["before"],
                        after=data["after"]
                    )
                else:
                    raise ValidationError(f"Unknown test type: {data['type']}")
                    
                tests.append(test)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_number}: {e}")
            except ValidationError as e:
                print(f"Validation error on line {line_number}: {e}")
            except KeyError as e:
                print(f"Missing required field on line {line_number}: {e}")
            except Exception as e:
                print(f"Unexpected error on line {line_number}: {e}")
                
    return tests


def save_tests(tests: list[BasePDFTest], jsonl_file: str) -> None:
    """Save tests to a JSONL file"""
    with open(jsonl_file, 'w') as file:
        for test in tests:
            # Convert dataclass to dict
            if isinstance(test, TextPresenceTest):
                data = {
                    "pdf": test.pdf,
                    "id": test.id,
                    "type": test.type,
                    "threshold": test.threshold,
                    "text": test.text
                }
            elif isinstance(test, TextOrderTest):
                data = {
                    "pdf": test.pdf,
                    "id": test.id,
                    "type": test.type,
                    "threshold": test.threshold,
                    "before": test.before,
                    "after": test.after
                }
            file.write(json.dumps(data) + '\n')
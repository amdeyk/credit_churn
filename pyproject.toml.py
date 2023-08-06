[build-system]
requires = [
    "setuptools>=42",
    "wheel"
]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
minversion = "2.0"
addopts = "-rfEX -p pytester --strict-markers"
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test", "Acceptance"]
python_functions = ["test"]
testpaths = ["tests"]
xfail_strict = true
filterwarnings = [
    "error",
    "default:Using or importing the ABCs:DeprecationWarning:unittest2.*",
    "default:Using or importing the ABCs:DeprecationWarning:pyparsing.*",
    "default:the imp module is deprecated in favour of importlib:DeprecationWarning:nose.*",
    "ignore:The distutils package is deprecated:DeprecationWarning",
    "ignore:.*U.*mode is deprecated:DeprecationWarning:(?!(pytest|_pytest))",
    "ignore:.*type argument to addoption.*:DeprecationWarning",
    "ignore:.*inspect.getargspec.*deprecated, use inspect.signature.*:DeprecationWarning",
    "ignore::pytest.PytestExperimentalApiWarning",
    "default:invalid escape sequence:DeprecationWarning",
    "ignore::_pytest.warning_types.PytestUnknownMarkWarning",
]

[tool.black]
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 100
lines_between_sections = 1
skip = "migrations"

### Running the tests

First, create a virtual environment and install the requirements with:

```console
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

<br>

To execute the tests, simply run from the root directory:

```console
pytest
```
<br>

Don't forget to also test the Transformers version with:

```console
pytest --transformers
```
<br>

Currently, there are several non-passing sentences. These will be marked with
`XFAIL`. If any of these sentences unexpectedly passed, it would be marked with
`XPASS`. In this case, please move the sentence from `tests.data.failing` to
its corresponding category (e.g., `tests.data.aux_root_negative`).

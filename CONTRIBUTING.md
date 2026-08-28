# Contributing

Contributions that improve correctness, reproducibility, documentation or
portability are welcome.

Before submitting a change:

1. create a branch from the current main branch;
2. keep changes focused and avoid adding generated simulation outputs;
3. run `python -m pytest`;
4. run `python Runner.py Example` for changes affecting the calculation path;
5. document changes that alter inputs, outputs or methodological assumptions.

Scientific-method changes should include a concise explanation of the physical
or statistical rationale and, when applicable, a reference or verification
case.

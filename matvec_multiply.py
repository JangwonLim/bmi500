"""Pure-Python matrix-vector multiplication.

Implements the dot product of two vectors and, on top of it, the product of a
matrix with a vector, using explicit loops only (no NumPy). A small benchmark
in ``main`` exercises the code on a randomly generated 1000x1000 problem.

Both public functions are defensive: they reject malformed input with a clear
message instead of failing deep inside the arithmetic, and they refuse to
return a silently wrong number. The failure modes guarded against are:

* Wrong container type -- a scalar, a generator, a string, or a flat list
  passed where a vector or a matrix of rows was expected.
* Non-numeric entries -- ``None``, strings, or objects that happen to support
  ``*`` (``"ab" * 2`` is a string, not an error, so this must be caught early).
* Non-finite entries -- ``nan`` and ``inf`` inputs, which otherwise propagate
  through the sum and yield a meaningless result without any warning.
* Overflow -- integers too large to convert to float, and sums of finite
  floats that overflow to ``inf``.
* Shape mismatch -- unequal vector lengths, ragged matrix rows, and a vector
  whose length differs from the matrix's column count.
* Loss of precision -- a naive running sum drops low-order bits when terms
  differ wildly in magnitude, so the accumulation is compensated.
"""

import math
import random
import time
from collections.abc import Mapping


def _check_is_sequence(candidate: object, description: str) -> None:
    """Raise TypeError unless ``candidate`` is a finite, indexable sequence.

    Strings, bytes and mappings are rejected explicitly: they are indexable and
    sized, so they would pass a structural check and then either fail
    confusingly during the arithmetic or, for a dict keyed by 0, 1, 2, ...,
    silently produce a number. Generators are rejected because they have no
    length and would be consumed by a single pass.

    Args:
        candidate: The object to check.
        description: Human-readable name used in the error message.

    Raises:
        TypeError: If the object cannot be used as a vector or a matrix row.
    """
    if isinstance(candidate, (str, bytes, bytearray)):
        raise TypeError(
            f"{description} must be a sequence of numbers, not "
            f"{type(candidate).__name__}."
        )
    if isinstance(candidate, Mapping):
        # A dict keyed by 0, 1, 2, ... is sized and indexable, so it would pass
        # the structural check below and quietly produce a number.
        raise TypeError(
            f"{description} must be a sequence of numbers, not a mapping."
        )
    if not hasattr(candidate, "__len__") or not hasattr(candidate, "__getitem__"):
        raise TypeError(
            f"{description} must be an indexable sequence with a length "
            f"(such as a list or tuple), got {type(candidate).__name__}."
        )


def _check_numeric_entries(vector, description: str) -> None:
    """Raise unless every entry of ``vector`` is a finite real number.

    Catching this up front turns two silent hazards into loud errors: entries
    that are not numbers at all, and entries that are ``nan`` or ``inf`` and
    would contaminate the result without any indication that it is garbage.

    Args:
        vector: An already shape-checked sequence.
        description: Human-readable name used in the error messages.

    Raises:
        TypeError: If an entry is not a real number.
        ValueError: If an entry is not finite, or is an integer too large to
            be represented as a float.
    """
    for index, entry in enumerate(vector):
        if isinstance(entry, complex) or not isinstance(entry, (int, float)):
            raise TypeError(
                f"{description}[{index}] must be a real number, got "
                f"{type(entry).__name__}."
            )
        try:
            is_finite = math.isfinite(entry)
        except OverflowError:
            # A Python int with no float representation, e.g. 10 ** 400.
            raise ValueError(
                f"{description}[{index}] is too large to convert to a float."
            ) from None
        if not is_finite:
            raise ValueError(
                f"{description}[{index}] must be finite, got {entry!r}."
            )


def dot_product(first_vector, second_vector) -> float:
    """Return the dot product of two equally sized vectors.

    The dot product is the sum of the element-wise products of the two
    vectors, i.e. ``sum(a[i] * b[i] for i in range(n))``.

    Args:
        first_vector: Sequence of finite real numbers of length n.
        second_vector: Sequence of finite real numbers of the same length n.

    Returns:
        The scalar dot product of the two vectors.

    Raises:
        TypeError: If either argument is not an indexable sequence, or holds a
            non-numeric entry.
        ValueError: If the vectors differ in length, or hold a non-finite or
            unrepresentably large entry.
        OverflowError: If the result overflows the range of a float.
    """
    _check_is_sequence(first_vector, "first_vector")
    _check_is_sequence(second_vector, "second_vector")

    if len(first_vector) != len(second_vector):
        raise ValueError(
            "Vectors must have the same length, got "
            f"{len(first_vector)} and {len(second_vector)}."
        )

    _check_numeric_entries(first_vector, "first_vector")
    _check_numeric_entries(second_vector, "second_vector")

    # Neumaier compensated summation: the running sum tracks the low-order
    # bits it would otherwise discard when terms differ greatly in magnitude,
    # and adds them back at the end. A naive sum of [1e16] + [0.1] * 1000 loses
    # the entire contribution of the small terms; this does not.
    running_total = 0.0
    lost_low_order_bits = 0.0

    for index in range(len(first_vector)):
        term = first_vector[index] * second_vector[index]
        new_total = running_total + term
        # Whichever operand is larger in magnitude keeps its bits; recover the
        # bits lost from the smaller one.
        if abs(running_total) >= abs(term):
            lost_low_order_bits += (running_total - new_total) + term
        else:
            lost_low_order_bits += (term - new_total) + running_total
        running_total = new_total

    result = running_total + lost_low_order_bits

    # Finite inputs can still sum past the float range; never hand back "inf"
    # as if it were an answer.
    if not math.isfinite(result):
        raise OverflowError(
            "Dot product overflowed the range of a float; the input values "
            "are too large."
        )

    return result


def matrix_vector_product(matrix, vector) -> list[float]:
    """Multiply a matrix by a vector.

    For a matrix of shape (num_rows, num_columns) and a vector of length
    num_columns, the result is a vector of length num_rows whose i-th entry is
    the dot product of the matrix's i-th row with the vector.

    The per-row arithmetic is delegated to ``dot_product``, so both functions
    share one definition of the sum and one set of entry checks. Structural
    problems (a non-sequence row, ragged rows, a column-count mismatch) are
    caught here first, before any arithmetic starts.

    Args:
        matrix: Sequence of rows, each row a sequence of finite real numbers of
            equal length.
        vector: Sequence of finite real numbers whose length matches the number
            of columns.

    Returns:
        The resulting vector, with one entry per row of the matrix.

    Raises:
        TypeError: If the matrix is not a sequence of row sequences, if the
            vector is not a sequence, or if any entry is non-numeric.
        ValueError: If the rows are ragged, if the column count does not match
            the vector length, or if any entry is non-finite or unrepresentably
            large.
        OverflowError: If any entry of the result overflows the float range.
    """
    _check_is_sequence(matrix, "matrix")
    _check_is_sequence(vector, "vector")
    _check_numeric_entries(vector, "vector")

    # An empty matrix has no rows to multiply, so the result is empty. The
    # vector is still validated above, so a malformed one is reported either
    # way rather than being silently ignored.
    if len(matrix) == 0:
        return []

    # Each row must be a sequence of numbers, and all rows must be the same
    # length as each other and as the vector, for the products to be defined.
    for row_index, row in enumerate(matrix):
        _check_is_sequence(row, f"matrix[{row_index}]")

    num_columns = len(matrix[0])
    for row_index, row in enumerate(matrix):
        if len(row) != num_columns:
            raise ValueError(
                f"All matrix rows must have the same length; row {row_index} "
                f"has length {len(row)}, expected {num_columns}."
            )

    if num_columns != len(vector):
        raise ValueError(
            f"Matrix has {num_columns} columns but vector has length "
            f"{len(vector)}."
        )

    # Each output entry is the dot product of one row with the input vector.
    # dot_product re-checks the entries of each row, so a bad entry is caught
    # there; the row is named again here to keep the message specific.
    result_vector = []
    for row_index, row in enumerate(matrix):
        try:
            result_vector.append(dot_product(row, vector))
        except (TypeError, ValueError, OverflowError) as error:
            # dot_product reports problems against its own parameter names;
            # rewrite them so the message points at the caller's data. Messages
            # that name neither parameter get the row index prepended instead.
            message = (
                str(error)
                .replace("first_vector", f"matrix[{row_index}]")
                .replace("second_vector", "vector")
            )
            if message == str(error):
                message = f"in matrix row {row_index}: {message}"
            raise type(error)(message) from error

    return result_vector


def main() -> None:
    """Benchmark the matrix-vector product on random 1000x1000 data.

    Generates a random 1000x1000 matrix and a random vector of length 1000,
    multiplies them, times the multiplication, and prints a short summary along
    with a spot check of the first entry against an independent computation.
    """
    matrix_size = 1000
    random.seed(0)  # Fixed seed so the benchmark is reproducible.

    # Random matrix (1000 rows x 1000 columns) and random vector (length 1000).
    matrix = [
        [random.random() for _ in range(matrix_size)] for _ in range(matrix_size)
    ]
    vector = [random.random() for _ in range(matrix_size)]

    start_time = time.perf_counter()
    result_vector = matrix_vector_product(matrix, vector)
    elapsed_seconds = time.perf_counter() - start_time

    print(f"Matrix shape: {len(matrix)}x{len(matrix[0])}")
    print(f"Vector length: {len(vector)}")
    print(f"Result length: {len(result_vector)}")
    print(f"Elapsed time: {elapsed_seconds:.3f} s")
    print(f"First result entry: {result_vector[0]:.6f}")

    # Spot check: recompute the first entry directly from the definition.
    expected_first_entry = math.fsum(a * b for a, b in zip(matrix[0], vector))
    print(
        "Matches direct computation: "
        f"{math.isclose(result_vector[0], expected_first_entry, rel_tol=1e-12)}"
    )


if __name__ == "__main__":
    main()

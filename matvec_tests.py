"""Unit tests for the functions in matvec_multiply.py.

Covers the dot-product and matrix-vector-product routines: known-value cases,
algebraic identities, edge cases such as empty and single-element inputs, and
the shape validation that makes both functions raise ValueError.

Run with:
    python -m unittest matvec_tests -v
"""

import contextlib
import io
import math
import random
import unittest

import bmi500.matvec_multiply as matvec_multiply
from bmi500.matvec_multiply import dot_product, matrix_vector_product


class DotProductTests(unittest.TestCase):
    """Tests for the dot_product function."""

    def test_known_value(self):
        """The dot product of [1, 2, 3] and [4, 5, 6] is 1*4 + 2*5 + 3*6 = 32."""
        self.assertAlmostEqual(dot_product([1, 2, 3], [4, 5, 6]), 32.0)

    def test_orthogonal_vectors_give_zero(self):
        """Perpendicular vectors have a dot product of zero."""
        self.assertAlmostEqual(dot_product([1, 0], [0, 1]), 0.0)

    def test_zero_vector_gives_zero(self):
        """Any vector dotted with the zero vector is zero."""
        self.assertAlmostEqual(dot_product([3.5, -2.0, 7.25], [0, 0, 0]), 0.0)

    def test_negative_and_float_values(self):
        """Negative and fractional entries are handled correctly."""
        # (-1.5 * 2.0) + (4.0 * -0.5) = -3.0 + -2.0 = -5.0
        self.assertAlmostEqual(dot_product([-1.5, 4.0], [2.0, -0.5]), -5.0)

    def test_single_element_vectors(self):
        """A one-element dot product is just the product of the two entries."""
        self.assertAlmostEqual(dot_product([6], [7]), 42.0)

    def test_empty_vectors_give_zero(self):
        """The dot product of two empty vectors is the empty sum, zero."""
        self.assertAlmostEqual(dot_product([], []), 0.0)

    def test_is_commutative(self):
        """Swapping the arguments does not change the result."""
        first_vector = [1.25, -3.0, 0.5, 8.0]
        second_vector = [2.0, 4.5, -1.0, 0.25]
        self.assertAlmostEqual(
            dot_product(first_vector, second_vector),
            dot_product(second_vector, first_vector),
        )

    def test_matches_builtin_sum_on_random_data(self):
        """Results agree with an independent sum/zip computation."""
        random.seed(1)
        first_vector = [random.uniform(-10, 10) for _ in range(100)]
        second_vector = [random.uniform(-10, 10) for _ in range(100)]
        expected_value = sum(a * b for a, b in zip(first_vector, second_vector))
        self.assertAlmostEqual(
            dot_product(first_vector, second_vector), expected_value
        )

    def test_mismatched_lengths_raise_value_error(self):
        """Vectors of different lengths have no defined dot product."""
        with self.assertRaises(ValueError):
            dot_product([1, 2, 3], [1, 2])

    def test_does_not_modify_inputs(self):
        """The inputs are read-only; the caller's lists stay unchanged."""
        first_vector = [1, 2, 3]
        second_vector = [4, 5, 6]
        dot_product(first_vector, second_vector)
        self.assertEqual(first_vector, [1, 2, 3])
        self.assertEqual(second_vector, [4, 5, 6])


class MatrixVectorProductTests(unittest.TestCase):
    """Tests for the matrix_vector_product function."""

    def test_known_value(self):
        """[[1, 2], [3, 4]] times [1, 1] is [1+2, 3+4] = [3, 7]."""
        result_vector = matrix_vector_product([[1, 2], [3, 4]], [1, 1])
        self.assertEqual(len(result_vector), 2)
        self.assertAlmostEqual(result_vector[0], 3.0)
        self.assertAlmostEqual(result_vector[1], 7.0)

    def test_identity_matrix_returns_the_vector(self):
        """Multiplying by the identity matrix leaves the vector unchanged."""
        identity_matrix = [
            [1 if row == column else 0 for column in range(3)] for row in range(3)
        ]
        vector = [2.5, -4.0, 7.0]
        result_vector = matrix_vector_product(identity_matrix, vector)
        for actual_entry, expected_entry in zip(result_vector, vector):
            self.assertAlmostEqual(actual_entry, expected_entry)

    def test_non_square_matrix(self):
        """A 2x3 matrix times a length-3 vector gives a length-2 result."""
        matrix = [[1, 2, 3], [4, 5, 6]]
        vector = [1, 0, -1]
        # Row 0: 1 - 3 = -2.  Row 1: 4 - 6 = -2.
        result_vector = matrix_vector_product(matrix, vector)
        self.assertEqual(len(result_vector), 2)
        self.assertAlmostEqual(result_vector[0], -2.0)
        self.assertAlmostEqual(result_vector[1], -2.0)

    def test_single_row_matrix_behaves_like_dot_product(self):
        """A one-row matrix yields a one-entry result equal to the dot product."""
        row = [1.5, 2.5, -3.0]
        vector = [2.0, -1.0, 4.0]
        result_vector = matrix_vector_product([row], vector)
        self.assertEqual(len(result_vector), 1)
        self.assertAlmostEqual(result_vector[0], dot_product(row, vector))

    def test_single_column_matrix_scales_the_rows(self):
        """A one-column matrix times a length-1 vector scales each row entry."""
        result_vector = matrix_vector_product([[2], [3], [4]], [5])
        self.assertEqual(len(result_vector), 3)
        for actual_entry, expected_entry in zip(result_vector, [10.0, 15.0, 20.0]):
            self.assertAlmostEqual(actual_entry, expected_entry)

    def test_zero_matrix_gives_zero_vector(self):
        """A matrix of zeros maps every vector to the zero vector."""
        zero_matrix = [[0, 0, 0], [0, 0, 0]]
        result_vector = matrix_vector_product(zero_matrix, [1.0, -2.0, 3.0])
        for entry in result_vector:
            self.assertAlmostEqual(entry, 0.0)

    def test_empty_matrix_gives_empty_result(self):
        """A matrix with no rows produces a result vector with no entries."""
        self.assertEqual(matrix_vector_product([], [1, 2, 3]), [])

    def test_result_length_equals_row_count(self):
        """The result has exactly one entry per row of the matrix."""
        random.seed(2)
        num_rows, num_columns = 7, 4
        matrix = [
            [random.random() for _ in range(num_columns)] for _ in range(num_rows)
        ]
        vector = [random.random() for _ in range(num_columns)]
        self.assertEqual(len(matrix_vector_product(matrix, vector)), num_rows)

    def test_matches_row_wise_reference_on_random_data(self):
        """Each entry matches an independent row-by-row computation."""
        random.seed(3)
        num_rows, num_columns = 20, 15
        matrix = [
            [random.uniform(-5, 5) for _ in range(num_columns)]
            for _ in range(num_rows)
        ]
        vector = [random.uniform(-5, 5) for _ in range(num_columns)]
        result_vector = matrix_vector_product(matrix, vector)
        for row, actual_entry in zip(matrix, result_vector):
            expected_entry = sum(a * b for a, b in zip(row, vector))
            self.assertAlmostEqual(actual_entry, expected_entry)

    def test_is_linear_in_the_vector(self):
        """A*(u + v) equals A*u + A*v, the defining property of a linear map."""
        matrix = [[1, 2], [3, 4], [5, 6]]
        first_vector = [1.0, 2.0]
        second_vector = [-3.0, 0.5]
        summed_vector = [a + b for a, b in zip(first_vector, second_vector)]

        product_of_sum = matrix_vector_product(matrix, summed_vector)
        sum_of_products = [
            a + b
            for a, b in zip(
                matrix_vector_product(matrix, first_vector),
                matrix_vector_product(matrix, second_vector),
            )
        ]
        for actual_entry, expected_entry in zip(product_of_sum, sum_of_products):
            self.assertAlmostEqual(actual_entry, expected_entry)

    def test_ragged_matrix_raises_value_error(self):
        """Rows of differing lengths do not form a valid matrix.

        The message must identify the offending row, so that a ragged matrix is
        distinguishable from a plain row/vector length mismatch.
        """
        with self.assertRaises(ValueError) as caught:
            matrix_vector_product([[1, 2], [3]], [1, 2])
        # Wording only the up-front ragged check produces; a length complaint
        # leaking out of the per-row dot product would not say this.
        self.assertIn("All matrix rows", str(caught.exception))
        self.assertIn("row 1 has length 1", str(caught.exception))
        self.assertIn("expected 2", str(caught.exception))

    def test_column_count_mismatch_raises_value_error(self):
        """The vector length must equal the number of matrix columns.

        The message must report the shapes, rather than surfacing a per-row
        length complaint from the underlying dot product.
        """
        with self.assertRaises(ValueError) as caught:
            matrix_vector_product([[1, 2, 3], [4, 5, 6]], [1, 2])
        self.assertIn("3 columns", str(caught.exception))
        self.assertIn("length 2", str(caught.exception))

    def test_does_not_modify_inputs(self):
        """The matrix and vector are read-only; the caller's data stays intact."""
        matrix = [[1, 2], [3, 4]]
        vector = [5, 6]
        matrix_vector_product(matrix, vector)
        self.assertEqual(matrix, [[1, 2], [3, 4]])
        self.assertEqual(vector, [5, 6])


class InputGuardTests(unittest.TestCase):
    """Tests for the defensive checks that reject malformed input.

    Each case covers a failure mode that would otherwise either crash deep in
    the arithmetic with an unhelpful message or, worse, return a number that
    looks plausible but is wrong.
    """

    def test_non_sequence_vector_raises_type_error(self):
        """A scalar is not a vector."""
        with self.assertRaises(TypeError):
            dot_product(5, [1])

    def test_generator_vector_raises_type_error(self):
        """A generator has no length and would be consumed by one pass."""
        with self.assertRaises(TypeError):
            dot_product((value for value in [1, 2]), [1, 2])

    def test_string_vector_raises_type_error(self):
        """A string is indexable but is not a vector of numbers."""
        with self.assertRaises(TypeError):
            dot_product("abc", [1, 2, 3])

    def test_non_numeric_entry_raises_type_error(self):
        """None is not a number, even though the container looks right."""
        with self.assertRaises(TypeError):
            dot_product([None], [1])

    def test_string_entry_raises_type_error(self):
        """A string entry must be caught: 'ab' * 2 is a string, not an error."""
        with self.assertRaises(TypeError):
            dot_product(["ab"], [2])

    def test_complex_entry_raises_type_error(self):
        """Complex numbers are outside the real-valued contract."""
        with self.assertRaises(TypeError):
            dot_product([1j], [1])

    def test_nan_entry_raises_value_error(self):
        """NaN would silently contaminate the whole sum."""
        with self.assertRaises(ValueError):
            dot_product([float("nan"), 1.0], [1.0, 1.0])

    def test_infinite_entry_raises_value_error(self):
        """Infinity would silently propagate into the result."""
        with self.assertRaises(ValueError):
            dot_product([float("inf")], [1.0])

    def test_unrepresentable_integer_raises_value_error(self):
        """An int with no float representation is reported clearly."""
        with self.assertRaises(ValueError):
            dot_product([10 ** 400], [1])

    def test_overflowing_sum_raises_overflow_error(self):
        """Finite inputs whose sum leaves the float range must not return inf."""
        with self.assertRaises(OverflowError):
            dot_product([1e308, 1e308], [1.0, 1.0])

    def test_flat_list_as_matrix_raises_type_error(self):
        """A flat list of numbers is not a matrix of rows."""
        with self.assertRaises(TypeError):
            matrix_vector_product([1, 2, 3], [1, 2, 3])

    def test_non_sequence_row_raises_type_error(self):
        """Every row must itself be a sequence."""
        with self.assertRaises(TypeError):
            matrix_vector_product([[1, 2], None], [1, 2])

    def test_non_numeric_matrix_entry_raises_type_error(self):
        """A bad entry anywhere in the matrix is caught before arithmetic."""
        with self.assertRaises(TypeError):
            matrix_vector_product([[1, 2], [3, "x"]], [1, 2])

    def test_nan_matrix_entry_raises_value_error(self):
        """A non-finite matrix entry is rejected."""
        with self.assertRaises(ValueError):
            matrix_vector_product([[1.0, float("nan")]], [1.0, 1.0])

    def test_empty_matrix_still_validates_the_vector(self):
        """The empty-matrix shortcut must not skip checking the vector."""
        with self.assertRaises(ValueError):
            matrix_vector_product([], [1.0, float("nan")])
        with self.assertRaises(TypeError):
            matrix_vector_product([], "abc")

    def test_compensated_summation_preserves_small_terms(self):
        """A naive running sum would drop the small terms entirely here.

        Summing 1e16 followed by a thousand 0.1 terms left to right loses every
        small term to rounding; the exact total is 1e16 + 100.
        """
        first_vector = [1e16] + [0.1] * 1000
        second_vector = [1.0] * 1001
        exact_total = math.fsum(
            a * b for a, b in zip(first_vector, second_vector)
        )
        self.assertEqual(dot_product(first_vector, second_vector), exact_total)

    def test_accepts_tuples_as_well_as_lists(self):
        """The guards check structure, not the exact container type."""
        self.assertAlmostEqual(dot_product((1, 2, 3), (4, 5, 6)), 32.0)
        self.assertEqual(matrix_vector_product(((1, 2), (3, 4)), (1, 1)), [3.0, 7.0])


class ContractAndMainTests(unittest.TestCase):
    """Tests for the module contract and for the main entry point.

    The first test pins the structural requirement that the matrix-vector
    product is built on top of dot_product rather than reimplementing the sum.
    The rest cover main, which the other test classes never exercise.
    """

    def test_matrix_vector_product_delegates_to_dot_product(self):
        """matrix_vector_product must compute each row via dot_product."""
        recorded_calls = []
        original_dot_product = matvec_multiply.dot_product

        def recording_dot_product(first_vector, second_vector):
            recorded_calls.append((list(first_vector), list(second_vector)))
            return original_dot_product(first_vector, second_vector)

        matvec_multiply.dot_product = recording_dot_product
        try:
            matvec_multiply.matrix_vector_product([[1, 2], [3, 4]], [1, 1])
        finally:
            matvec_multiply.dot_product = original_dot_product

        # One call per row, each pairing that row with the input vector.
        self.assertEqual(
            recorded_calls, [([1, 2], [1, 1]), ([3, 4], [1, 1])]
        )

    def test_row_error_message_names_the_offending_row(self):
        """A bad entry is reported against the matrix, not a parameter name."""
        with self.assertRaises(TypeError) as caught:
            matrix_vector_product([[1, 2], [3, "x"]], [1, 2])
        self.assertIn("matrix[1][1]", str(caught.exception))

    def test_row_overflow_is_reported_with_its_row_index(self):
        """An overflow names no parameter, so the row index is prepended."""
        with self.assertRaises(OverflowError) as caught:
            matrix_vector_product([[1.0, 1.0], [1e308, 1e308]], [1.0, 1.0])
        self.assertIn("matrix row 1", str(caught.exception))

    def test_mapping_is_rejected(self):
        """A dict keyed by 0, 1, ... is sized and indexable but is not a vector."""
        with self.assertRaises(TypeError):
            dot_product({0: 1, 1: 2}, [1, 2])
        with self.assertRaises(TypeError):
            matrix_vector_product({0: [1, 2]}, [1, 2])

    def test_bytes_are_rejected(self):
        """Bytes are indexable and yield ints, so they must be excluded."""
        with self.assertRaises(TypeError):
            dot_product(b"ab", [1, 2])

    def test_zero_column_matrix_gives_zero_entries(self):
        """Rows with no columns produce empty sums, one zero per row."""
        self.assertEqual(matrix_vector_product([[]], []), [0.0])
        self.assertEqual(matrix_vector_product([[], []], []), [0.0, 0.0])

    def test_main_runs_and_reports_a_consistent_result(self):
        """main completes on 1000x1000 data and its self-check passes."""
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            matvec_multiply.main()
        printed_text = captured_output.getvalue()

        self.assertIn("Matrix shape: 1000x1000", printed_text)
        self.assertIn("Vector length: 1000", printed_text)
        self.assertIn("Result length: 1000", printed_text)
        # main prints the outcome of its own spot check; it must not be False.
        self.assertIn("Matches direct computation: True", printed_text)


if __name__ == "__main__":
    unittest.main()

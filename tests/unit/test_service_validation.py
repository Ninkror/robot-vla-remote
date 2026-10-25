import unittest

import numpy as np

from service.validation import decode_rgb, validate_state


class ValidateStateTests(unittest.TestCase):
    def test_accepts_exactly_six_finite_values_as_independent_float32(self):
        source = [0, 1, 2, 3, 4, 5]

        result = validate_state(source)

        np.testing.assert_array_equal(result, np.arange(6, dtype=np.float32))
        self.assertEqual(result.dtype, np.float32)
        self.assertFalse(result.flags.writeable)
        source[0] = 99
        self.assertEqual(float(result[0]), 0.0)

    def test_rejects_wrong_length(self):
        with self.assertRaisesRegex(ValueError, "exactly 6"):
            validate_state([0] * 5)

    def test_rejects_non_finite_value(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_state([0, 1, 2, 3, 4, float("nan")])


class DecodeRgbTests(unittest.TestCase):
    def test_decodes_rgb8_with_padded_rows(self):
        data = bytes([1, 2, 3, 4, 5, 6, 99, 99])

        result = decode_rgb(1, 2, "rgb8", 8, data)

        np.testing.assert_array_equal(
            result,
            np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8),
        )
        self.assertTrue(result.flags.c_contiguous)
        self.assertFalse(result.flags.writeable)

    def test_converts_bgr8_to_rgb(self):
        result = decode_rgb(1, 1, "bgr8", 3, bytes([10, 20, 30]))

        np.testing.assert_array_equal(result, np.array([[[30, 20, 10]]], dtype=np.uint8))

    def test_rejects_unknown_encoding(self):
        with self.assertRaisesRegex(ValueError, "encoding"):
            decode_rgb(1, 1, "mono8", 1, bytes([0]))

    def test_rejects_buffer_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "buffer length"):
            decode_rgb(1, 1, "rgb8", 3, bytes([0, 1]))


if __name__ == "__main__":
    unittest.main()

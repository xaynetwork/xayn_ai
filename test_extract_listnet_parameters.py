#!/usr/bin/python3

from io import BytesIO
import unittest
import numpy as np

from extract_listnet_parameters import Bincode, write_list_net_parameters_to_encoder

class TestEncodeMatrices(unittest.TestCase):

    def test_encode_matrices(self):
        io = BytesIO(bytearray())
        encoder = Bincode(output=io)
        # Note: This test is unstable with python <v3.7.
        matrices = {
            'a': np.array([[1., 2.], [3., 4.]], dtype=np.float32),
            'b': np.array([3., 2., 1., 4.], dtype=np.float32)
        }
        write_list_net_parameters_to_encoder(encoder, matrices)
        output = io.getvalue()
        self.assertEqual(output, b'\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00a\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@\x01\x00\x00\x00\x00\x00\x00\x00b\x01\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00@@\x00\x00\x00@\x00\x00\x80?\x00\x00\x80@')

class TestBincode(unittest.TestCase):

    def setUp(self):
        self.io = BytesIO(bytearray())
        self.encoder = Bincode(output=self.io)

    def assertIoEqual(self, expected: bytes):
        data = self.io.getvalue()
        self.assertEqual(data, expected)

    def test_write_usize_in_correct_byteorder(self):
        self.encoder.write_usize(0xFCFA)
        self.assertIoEqual(b'\xFA\xFC\x00\x00\x00\x00\x00\x00')

        io = BytesIO(bytearray())
        encoder = Bincode(output=io, byteorder=Bincode.BIG_ENDIAN)
        encoder.write_usize(0xFCFA)
        self.assertEqual(io.getvalue(), b'\x00\x00\x00\x00\x00\x00\xFC\xFA')


    def test_write_string(self):
        input = "😀"
        expected = b'\x04\x00\x00\x00\x00\x00\x00\x00\xf0\x9f\x98\x80'
        self.encoder.write_string(input)
        self.assertIoEqual(expected)

    def test_write_byte_slice(self):
        input = b'\x01\xFF\x20'
        expected = b'\x03\x00\x00\x00\x00\x00\x00\x00\x01\xFF\x20'
        self.encoder.write_byte_slice(input)
        self.assertIoEqual(expected)

    def test_write_byte(self):
        input = 0x20
        expected = b'\x20'
        self.encoder.write_byte(input)
        self.assertIoEqual(expected)

    def test_write_map(self):
        # Note: This test is unstable with python <v3.7.
        input = { 200:  b'\x02\x00\x00', 32: b'\xFF\xCF' }
        expected = b'\x02\x00\x00\x00\x00\x00\x00\x00\xC8\x03\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x20\x02\x00\x00\x00\x00\x00\x00\x00\xFF\xCF'
        self.encoder.write_map(input, self.encoder.write_byte, self.encoder.write_byte_slice)
        self.assertIoEqual(expected)

    def test_write_list(self):
        input = [0xFC, 0xAB, 0x20, 0x30]
        expected = b'\x04\x00\x00\x00\x00\x00\x00\x00\xFC\xAB\x20\x30'
        self.encoder.write_list(input, self.encoder.write_byte)
        self.assertIoEqual(expected)

    def test_write_multi_dim_array(self):
        input = np.array([
            [[False, True], [True, True]],
            [[False, False],[False, True]],
            [[False, False],[True, True]]
        ], dtype=np.bool8)
        expected = b'\x03\x00\x00\x00\x00\x00\x00\x00' \
            b'\x03\x00\x00\x00\x00\x00\x00\x00' \
            b'\x02\x00\x00\x00\x00\x00\x00\x00' \
            b'\x02\x00\x00\x00\x00\x00\x00\x00' \
            b'\x0c\x00\x00\x00\x00\x00\x00\x00' \
            b'\x00\x01\x01\x01\x00\x00\x00\x01\x00\x00\x01\x01'
        self.encoder.write_array(input)
        self.assertIoEqual(expected)

    def test_write_array_little_endian(self):
        input = np.array([0xFFC, 0x1ABA], dtype=np.uint16)
        expected = b'\x01\x00\x00\x00\x00\x00\x00\x00' \
            b'\x02\x00\x00\x00\x00\x00\x00\x00' \
            b'\x02\x00\x00\x00\x00\x00\x00\x00' \
            b'\xFC\x0F\xBA\x1A'
        self.encoder.write_array(input)
        self.assertIoEqual(expected)

    def test_write_array_big_endian(self):
        input = np.array([0xFFC, 0x1ABA], dtype=np.uint16)
        expected = b'\x00\x00\x00\x00\x00\x00\x00\x01' \
            b'\x00\x00\x00\x00\x00\x00\x00\x02' \
            b'\x00\x00\x00\x00\x00\x00\x00\x02' \
            b'\x0F\xFC\x1A\xBA'
        io = BytesIO(bytearray())
        encoder = Bincode(output=io, byteorder=Bincode.BIG_ENDIAN)
        encoder.write_array(input)
        self.assertEqual(io.getvalue(), expected)

if __name__ == '__main__':
    unittest.main()
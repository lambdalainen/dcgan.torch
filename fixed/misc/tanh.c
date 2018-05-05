// Implementation of tanh: (1 - exp(-2x)) / (1 + exp(-2x)).

// Returns (1 - x) / (1 + x) for x in (0, 1).
template <typename tRawType>
FixedPoint<tRawType, 0> one_minus_x_over_one_plus_x_for_x_in_0_1(
    FixedPoint<tRawType, 0> a) {
  typedef FixedPoint<tRawType, 0> F0;
  typedef FixedPoint<tRawType, 2> F2;
  F0 half_denominator = RoundingHalfSum(a, F0::One());
  // Newton-Raphson division
  // https://en.wikipedia.org/wiki/Division_algorithm#Newton.E2.80.93Raphson_division
  // Refer to that page for the logic behind the 48/17 and 32/17 constants.
  const F2 constant_48_over_17 =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F2, 1515870810, 48.0 / 17.0);
  const F2 constant_neg_32_over_17 =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F2, -1010580540, -32.0 / 17.0);
  F2 x = constant_48_over_17 + half_denominator * constant_neg_32_over_17;
  for (int i = 0; i < 3; i++) {
    F2 half_denominator_times_x = half_denominator * x;
    F2 one_minus_half_denominator_times_x =
        F2::One() - half_denominator_times_x;
    x = x + Rescale<2>(x * one_minus_half_denominator_times_x);
  }
  return Rescale<0>(x - F2::One());
}

// Returns -tanh(x) for x < 0.
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> neg_tanh_on_negative_values(
    FixedPoint<tRawType, tIntegerBits> a) {
  return one_minus_x_over_one_plus_x_for_x_in_0_1(
      exp_on_negative_values(ExactMulByPot<1>(a)));
}

// Returns tanh(x) for any x.
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> tanh(FixedPoint<tRawType, tIntegerBits> a) {
  typedef FixedPoint<tRawType, tIntegerBits> InputF;
  typedef FixedPoint<tRawType, 0> ResultF;
  tRawType mask_if_negative = MaskIfLessThan(a, InputF::Zero());
  tRawType mask_if_zero = MaskIfZero(a);
  InputF n = SelectUsingMask(mask_if_negative, a, -a);
  ResultF t = neg_tanh_on_negative_values(n);
  return SelectUsingMask(mask_if_zero, ResultF::Zero(),
                         SelectUsingMask(mask_if_negative, -t, t));
}

# Evo-RLT adaptation

The RL-token encoder/decoder, reference-conditioned absolute chunk actor and
twin-Q objectives are adapted from MINT-SJTU/Evo-RLT (Apache-2.0), reference
commit `474c66958cffead65cbbf314ff85d0b85ef5f02f`:

https://github.com/MINT-SJTU/Evo-RLT

Source components: `src/evo_rlt/core/rl_token.py`, `actor.py`, `critic.py`,
`losses.py`, and the `RLTActionModifier` deployment boundary. The Apache-2.0
license text is included in this repository's root LICENSE.

This port avoids the upstream LeRobot 0.5.1/Python 3.12 adapter dependency. It
uses this repository's PI05 inference-time RTC and normalization, consumes the
full unfiltered VLA action reference, bypasses S-G whenever RLT is enabled,
protects the guided prefix during training and deployment,
and masks short terminal chunks. Its explicit checkpoint contract and network
configuration are local to this integration; upstream checkpoints are not
drop-in compatible. Actor output is an absolute normalized target, not an
additive action residual. BC uses the raw mean to preserve gradients outside
the clipped deployment range.

S-G and RLT are mutually exclusive deployment paths. RLT uses an effective S-G
window of zero; old S-G-reference caches and heads must be re-encoded/retrained
for the direct-VLA contract. Ordinary RTC can retain its existing S-G filter.

"""GDPVal-style benchmark comparing OpenSpace and Hermes, with and without Reflexio.

Runs the GDPVal dataset through two host agents (OpenSpace, Hermes) in a
three-phase protocol (cold → host-warm → host-warm + reflexio) so we can
isolate the marginal contribution of Reflexio's memory layer on top of each
host's own native learning system.
"""

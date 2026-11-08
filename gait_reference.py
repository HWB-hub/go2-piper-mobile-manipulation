"""Continuous foot references, in FL/FR/RL/RR order and SI units.

Inspired by Walk These Ways phase-shaped contacts and Raibert foot placement.
This module specifies rewards; it does not overwrite simulated joints or poses.
"""
import math
import torch


def equivalent_speed(velocity):
    """Combine translation [m/s] and rotation at a 0.30 m foot radius."""
    return velocity[:, :2].norm(dim=-1) + .30 * velocity[:, 2].abs()


def references(phase, frequency, velocity, running=False):
    """Return contact probability, sole clearance [m], foot xy [m], moving gate."""
    speed = equivalent_speed(velocity)
    moving = ((speed - .035) / .065).clamp(0., 1.)
    duty = (.60 - .07 * speed).clamp(.52, .60)
    if running:
        blend = ((speed - .5) / .85).clamp(0., 1.)
        blend = blend.square() * (3. - 2. * blend)
        duty = torch.lerp(duty, torch.full_like(duty, .38), blend)
    p = torch.remainder(phase[:, None] + phase.new_tensor([0., .5, .5, 0.]), 1.)
    stance = p < duty[:, None]
    swing = ((p - duty[:, None]) / (1. - duty[:, None])).clamp(0., 1.)
    warped = torch.where(stance, .5 * p / duty[:, None], .5 + .5 * swing)
    # Periodic smoothed stance window; both sides of phase wrap agree.
    def cdf(x):
        return .5 * (1. + torch.erf(x / (.04 * math.sqrt(2.))))
    contact = cdf(warped) * (1. - cdf(warped - .5)) + cdf(warped - 1.) * (1. - cdf(warped - 1.5))
    contact = 1. - moving[:, None] * (1. - contact)
    clearance = (.045 + .025 * speed).clamp(max=.065)
    height = clearance[:, None] * (64. * swing**3 * (1.-swing)**3) * moving[:, None]
    nominal = phase.new_tensor([[.20, .165], [.20, -.165], [-.23, .165], [-.23, -.165]])
    foot_velocity = velocity[:, None, :2] + velocity[:, 2, None, None] * torch.stack((-nominal[:, 1], nominal[:, 0]), dim=-1)
    half_stride = .5 * duty[:, None, None] / frequency[:, None, None] * foot_velocity
    # Match stance velocity and acceleration at both swing boundaries (C2).
    smooth = 10 * swing**3 - 15 * swing**4 + 6 * swing**5
    endpoint_slope = -(1.-duty[:,None])/duty[:,None]
    blend = (1.-endpoint_slope)*smooth + endpoint_slope*swing
    progress = torch.where(stance, 1. - 2. * p / duty[:, None], -1. + 2. * blend)
    xy = nominal[None] + half_stride * progress[:, :, None] * moving[:, None, None]
    return contact, height, xy, moving

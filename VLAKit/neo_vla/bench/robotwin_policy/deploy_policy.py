"""The plug-in RoboTwin's script/eval_policy.py loads as policy/neo_policy.

robotwin_run copies this directory into the RoboTwin checkout. Everything the
plug-in needs arrives in one settings file named by --neo_settings: the harness
runs each override through eval(), so a path or a url passed on its own could
arrive as something other than its text.
"""
from neo_vla.bench.robotwin import agent_from_settings


def encode_obs(observation):
    return observation


def get_model(usr_args):
    return agent_from_settings(usr_args["neo_settings"])


def eval(TASK_ENV, model, observation):  # noqa: A001 - the name the harness calls
    model.act(TASK_ENV, encode_obs(observation))


def reset_model(model):
    model.reset()

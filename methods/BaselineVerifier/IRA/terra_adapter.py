from . import judge


def _make_terra(env, app):
    from .IRA_Terra import IRATerra
    return IRATerra(env, app=app)


def verify_episode(env, instr, example_result_dir, **kw):
    return judge(env, instr, example_result_dir, _make_terra)

import pytest
import typer
from typer.testing import CliRunner

from mi_bici.config import MODEL_DEFAULTS
from mi_bici.modeling.train import app, parse_hyperparam_args, resolve_hyperparams


def test_parse_hyperparam_args_formats():
    args = ['--alpha', '0.5', '--max_depth=3']
    assert parse_hyperparam_args(args) == {'alpha': '0.5', 'max_depth': '3'}
    with pytest.raises(typer.BadParameter):
        parse_hyperparam_args(['--alpha'])
    with pytest.raises(typer.BadParameter):
        parse_hyperparam_args(['0.5'])


def test_resolve_hyperparams_uses_config_defaults():
    assert resolve_hyperparams('p1', 'ridge', {}) == MODEL_DEFAULTS['p1']['ridge']
    assert resolve_hyperparams('p1', 'ridge', {'alpha': '0.5'}) == {'alpha': 0.5}


@pytest.mark.parametrize(
    ('question', 'model', 'name'),
    [
        (question, model, name)
        for question, models in MODEL_DEFAULTS.items()
        for model, hparams in models.items()
        for name in hparams
    ],
)
def test_every_hyperparameter_is_accepted(question, model, name):
    default = MODEL_DEFAULTS[question][model][name]
    hparams = resolve_hyperparams(question, model, {name: str(default)})
    assert hparams[name] == default
    assert type(hparams[name]) is type(default)


def test_resolve_hyperparams_rejects_invalid():
    with pytest.raises(typer.BadParameter, match='Opciones'):
        resolve_hyperparams('p1', 'modelo_inexistente', {})
    with pytest.raises(typer.BadParameter, match='Hiperparámetros válidos: max_iter'):
        resolve_hyperparams('p2', 'hgb', {'alpha': '1'})
    with pytest.raises(typer.BadParameter, match='Valor inválido'):
        resolve_hyperparams('p2', 'hgb', {'max_depth': 'x'})


def test_list_models_shows_all_models():
    result = CliRunner().invoke(app, ['--list-models'])
    assert result.exit_code == 0
    for question, models in MODEL_DEFAULTS.items():
        assert question in result.output
        for model in models:
            assert model in result.output

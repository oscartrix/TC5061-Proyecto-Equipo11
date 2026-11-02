"""Ejecuta chequeos reales y conserva registros, entorno y capturas de su visor HTML."""

import argparse
from datetime import datetime, timezone
import hashlib
import html
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
# Orquestación de herramientas locales mediante listas de argumentos.
import subprocess  # nosec B404
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ['pytest', 'deepchecks', 'anywidget', 'setuptools', 'bandit', 'pip-audit']


def run_local(command, **kwargs):
    """Limita los ejecutables a Python, Git y navegadores resueltos localmente."""
    allowed = {sys.executable, shutil.which('git'), shutil.which('chromium'),
               shutil.which('google-chrome')} - {None}
    if command[0] not in allowed or kwargs.get('shell'):
        raise ValueError('El colector solo permite herramientas locales conocidas, sin shell.')
    # Ejecutable permitido y argumentos separados; nunca se usa un intérprete de shell.
    return subprocess.run(command, **kwargs)  # nosec B603


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def display_summary(name, output):
    """Resume salidas voluminosas sin sustituir el registro completo."""
    if name == '02_deepchecks_calidad':
        markers = ('===== VALIDANDO', 'Registros totales:', 'Columnas:',
                   'Filas duplicadas:', 'Valores faltantes totales:', 'Reporte guardado')
        return '\n'.join(line for line in output.splitlines() if any(m in line for m in markers))
    if name == '03_deepchecks_drift':
        import pandas as pd

        frame = pd.read_csv(ROOT / 'reports/data_drift/drift_summary.csv')
        flagged = frame.loc[frame.status != 'pass', [
            'dataset', 'feature', 'drift_score', 'warning_threshold', 'error_threshold', 'status',
        ]]
        return ('Resumen de reports/data_drift/drift_summary.csv\n\n'
                + frame['status'].value_counts().to_string() + '\n\n'
                + flagged.to_string(index=False, float_format=lambda value: f'{value:.4f}')
                + '\n\nCódigo 0: las advertencias permiten continuar según la política configurada.')
    if name == '04_seguridad':
        pillow = next(line for line in (ROOT / 'requirements.txt').read_text().splitlines()
                      if line.lower().startswith('pillow=='))
        return (f'bandit {version("bandit")} | pip-audit {version("pip-audit")}\n'
                f'Auditoría de requirements.txt completo, incluida la entrada {pillow}\n\n' + output)
    if name in {'05_evaluacion', '06_inferencia'}:
        import pandas as pd

        if name == '06_inferencia':
            return '\n'.join(line for line in output.splitlines()
                             if 'predicciones guardadas' in line)
        sections = []
        columns = {
            'p1': ['split', 'n_samples', 'model_mae', 'baseline_mae', 'mae_improvement_pct'],
            'p2': ['split', 'n_samples', 'model_f1_macro', 'baseline_f1_macro',
                   'model_missed_deficits', 'baseline_missed_deficits',
                   'model_false_deficit_alerts', 'baseline_false_deficit_alerts'],
            'p3': ['model', 'split', 'cohort', 'n_stations', 'silhouette_usage'],
        }
        for question in columns:
            frame = pd.read_csv(ROOT / 'reports/evaluation' / question / 'summary.csv')
            selected = frame[columns[question]].rename(columns={
                'n_samples': 'n', 'model_f1_macro': 'model_F1',
                'baseline_f1_macro': 'base_F1',
                'model_missed_deficits': 'model_missed',
                'baseline_missed_deficits': 'base_missed',
                'model_false_deficit_alerts': 'model_false',
                'baseline_false_deficit_alerts': 'base_false',
            })
            label = question.upper()
            if question == 'p2':
                label += ' (missed: déficits omitidos; false: falsas alertas)'
            sections.append(label + '\n' + selected.to_string(
                index=False, float_format=lambda value: f'{value:.4f}'
            ))
        return '\n\n'.join(sections)
    return output


def screenshot(browser, destination, title, record, summary, python_version):
    """Captura el registro renderizado por Chromium, sin generar resultados sintéticos."""
    commands = '\n'.join('$ ' + shlex.join(item['command']) for item in record['commands'])
    summary = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', summary)
    content = f'''<!doctype html><html lang="es"><meta charset="utf-8">
<title>{html.escape(title)}</title><style>
body {{ margin: 0; background: #10151c; color: #e4ebf4; padding: 30px;
font: 14px/1.5 "DejaVu Sans Mono", monospace; }}
h1 {{ font-size: 22px; color: #8fc9ff; margin: 0 0 12px; }}
p {{ margin: 6px 0; color: #b2c0d1; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; margin: 16px 0 0;
background: #19212c; padding: 18px; border: 1px solid #354358; }}
.status {{ color: {'#9de2af' if record['exit_code'] == 0 else '#ff9e9e'}; }}
</style><h1>{html.escape(title)}</h1>
<p>Visor de registros reales · Python {html.escape(python_version)}</p>
<p>Inicio UTC: {html.escape(record['started_at'])}</p>
<p>Fin UTC: {html.escape(record['finished_at'])}</p>
<p class="status">Código de salida: {record['exit_code']}</p>
<p>Resumen de salida / reportes. Registro completo: {html.escape(record['log_file'])}</p>
<pre>{html.escape(commands)}</pre><pre>{html.escape(summary)}</pre></html>'''
    # Altura suficiente para capturar todo el resumen, incluidas líneas largas ajustadas.
    lines = sum(max(1, (len(line) + 138) // 139) for line in (commands + '\n' + summary).splitlines())
    height = max(650, 450 + 22 * lines)
    with tempfile.TemporaryDirectory(prefix='mibici-evidence-') as temporary:
        page = Path(temporary) / 'evidence.html'
        page.write_text(content, encoding='utf-8')
        command = [
            browser, '--headless', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
            '--disable-crash-reporter', '--disable-breakpad',
            '--disable-background-networking', '--no-first-run', '--hide-scrollbars',
            '--force-device-scale-factor=1', f'--user-data-dir={temporary}/profile',
            f'--window-size=1280,{height}', f'--screenshot={destination}', page.as_uri(),
        ]
        browser_env = dict(os.environ, XDG_CONFIG_HOME=f'{temporary}/config',
                           XDG_CACHE_HOME=f'{temporary}/cache')
        result = run_local(command, capture_output=True, text=True, timeout=60, env=browser_env)
        if result.returncode or not destination.is_file():
            raise RuntimeError(f'No se pudo capturar el visor de registros: {result.stderr}')


def collect(output_dir, screenshots=False, include_models=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    browser = shutil.which('chromium') or shutil.which('google-chrome')
    if screenshots and not browser:
        raise ValueError('Las capturas requieren Chromium o Google Chrome instalado.')
    python = sys.executable
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MPLBACKEND='Agg', NO_COLOR='1')
    steps = [
        ('01_pytest', 'PyTest: suite completa', [[python, '-m', 'pytest', '-q', 'tests',
                                               '--disable-warnings']]),
        ('02_deepchecks_calidad', 'Contratos y calidad de datos',
         [[python, '-m', 'mi_bici.validation.run_validation']]),
        ('03_deepchecks_drift', 'Drift: scores, umbrales y estados',
         [[python, '-m', 'mi_bici.validation.drift_validation']]),
        ('04_seguridad', 'Seguridad: mi_bici/ y requirements.txt', [
            [python, '-m', 'bandit', '-r', 'mi_bici'],
            [python, '-m', 'pip_audit', '-r', 'requirements.txt', '--no-deps', '--disable-pip'],
        ]),
    ]
    if include_models:
        steps += [
            ('05_evaluacion', 'Champions frente a baselines', [
                [python, '-m', 'mi_bici.modeling.evaluate', '--question', question]
                for question in ('p1', 'p2', 'p3')
            ]),
            ('06_inferencia', 'Inferencia con los tres champions', [
                [python, '-m', 'mi_bici.modeling.predict', '--question', question,
                 '--features-path', f'data/processed/{source}.parquet',
                 '--predictions-path', f'data/processed/{question}_predictions.parquet']
                for question, source in [('p1', 'p1_features'), ('p2', 'p2_features'),
                                         ('p3', 'p3_holdout_features')]
            ]),
        ]
    metadata = {
        'started_at': utc_now(), 'python': platform.python_version(),
        'platform': platform.platform(), 'packages': {name: version(name) for name in PACKAGES},
        'git_commit': run_local([shutil.which('git'), 'rev-parse', 'HEAD'], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip(),
        'working_tree_status': run_local(
            [shutil.which('git'), 'status', '--porcelain'], cwd=ROOT,
            capture_output=True, text=True, check=True
        ).stdout.splitlines(),
        'requirements_sha256': {name: sha256(ROOT / name)
                                for name in ('requirements.txt', 'requirements-test.txt')},
        'steps': [],
    }
    freeze = run_local([python, '-m', 'pip', 'freeze'], cwd=ROOT, env=env,
                       capture_output=True, text=True, check=True).stdout
    (output_dir / 'environment.txt').write_text(freeze, encoding='utf-8')
    for name, title, commands in steps:
        print(f'INICIO {name}', flush=True)
        record = {'name': name, 'started_at': utc_now(), 'commands': [], 'exit_code': 0,
                  'log_file': f'{name}.txt'}
        transcript = []
        for command in commands:
            started_at = utc_now()
            result = run_local(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, timeout=600)
            record['commands'].append({'command': command, 'started_at': started_at,
                                       'finished_at': utc_now(), 'exit_code': result.returncode})
            transcript.append(f'$ {shlex.join(command)}\n{result.stdout}\n'
                              f'[exit_code={result.returncode}]\n')
            if result.returncode:
                record['exit_code'] = result.returncode
        record['finished_at'] = utc_now()
        output = '\n'.join(transcript)
        log = output_dir / record['log_file']
        log.write_text(output, encoding='utf-8')
        record['log_sha256'] = sha256(log)
        if screenshots:
            # Los fallos se muestran completos; nunca se ocultan detrás de un resumen positivo.
            summary = display_summary(name, output) if record['exit_code'] == 0 else output
            destination = output_dir / f'{name}.png'
            screenshot(browser, destination, title, record, summary, metadata['python'])
            record['screenshot_file'] = destination.name
            record['screenshot_sha256'] = sha256(destination)
        metadata['steps'].append(record)
        metadata['finished_at'] = utc_now()
        (output_dir / 'execution.json').write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        print(f'FIN {name}: exit_code={record["exit_code"]}', flush=True)
    return 1 if any(step['exit_code'] for step in metadata['steps']) else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'docs/evidencia')
    parser.add_argument('--screenshots', action='store_true', help='Capturar el visor con Chromium')
    parser.add_argument('--include-models', action='store_true', help='Evaluar e inferir con champions')
    options = parser.parse_args()
    raise SystemExit(collect(options.output_dir.resolve(), options.screenshots, options.include_models))

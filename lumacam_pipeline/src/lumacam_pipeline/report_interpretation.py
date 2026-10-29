"""Evidence-linked scientific interpretation supplied by a human or analysis job.

Report assembly verifies the cited files; it never invents scientific conclusions.
"""
import json
from pathlib import Path
import textwrap

from .handover import file_digest


def load_interpretation(output, roots):
    path = Path(output) / 'interpretation.json'
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported interpretation schema; expected 1.')
    expected = {(path.parent / name).resolve() for name in data.get('source_profiles', [])}
    stale = expected != set(roots) or not data.get('evidence')
    cited = set()
    for item in data.get('evidence', []):
        source = (path.parent / item['path']).resolve()
        cited.add(source)
        if not source.is_file() or file_digest(source) != item.get('sha256'):
            stale = True
    for page in data.get('comparison_pages', []):
        if (path.parent / page['figure']['path']).resolve() not in cited:
            stale = True
    if stale:
        return {'warning': 'Scientific interpretation needs review: source profiles or cited evidence changed. '
                           'Previous conclusions and comparison figures are withheld until interpretation.json is refreshed.'}
    for page in data.get('comparison_pages', []):
        page['figure']['path'] = (path.parent / page['figure']['path']).resolve()
        if not page['figure']['path'].is_file():
            raise ValueError('An interpretation figure is missing.')
    return data


def experiment_interpretation(data, report):
    return data.get('experiments', {}).get(report.root.name, {})


def interpretation_pdf_pages(data, title, save):
    """Conclusion callouts and comparison pages, drawn with the report backend."""
    if not data.get('summary'):
        return
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.image as mpimg

    def new(heading):
        fig = Figure(figsize=(8.27, 11.69))
        fig.text(.07, .96, '\n'.join(textwrap.wrap(heading, 66)), fontsize=12,
                 weight='bold', va='top')
        return fig

    def text(fig, value, y, *, label='', shade=False, size=9):
        lines = textwrap.wrap((label + ' ' + value).strip(), 103)
        height = .0175 * max(1, len(lines)) + .015
        if shade:
            fig.add_artist(FancyBboxPatch((.065, y-height), .87, height+.006,
                boxstyle='round,pad=0.006', facecolor='#eaf2f6', edgecolor='none',
                transform=fig.transFigure, zorder=0))
        fig.text(.077 if shade else .07, y-.004, '\n'.join(lines), fontsize=size,
                 va='top', color='#183b4d' if shade else '#222222')
        return y-height-.012

    fig = new(title); y = .89
    y = text(fig, data.get('scope', ''), y)
    for item in data['summary']:
        if y < .30:
            save(fig); fig = new(title + ' — conclusions'); y = .89
        fig.text(.07, y, item['heading'], fontsize=10, weight='bold', va='top'); y -= .027
        y = text(fig, item['observation'], y, label='Observation:')
        y = text(fig, item['conclusion'], y, label='Conclusion:', shade=True)
    if data.get('limitations'):
        if y < .19:
            save(fig); fig = new('Interpretation limits'); y = .89
        fig.text(.07, y, 'How to read these conclusions', fontsize=9, weight='bold', va='top'); y -= .026
        for line in data['limitations']:
            if y < .10:
                save(fig); fig = new('Interpretation limits'); y = .89
            y = text(fig, line, y, size=8)
    save(fig)
    for page in data.get('comparison_pages', []):
        fig = new(page['title'])
        text(fig, page['observation'], .89, label='Observation:')
        axis = fig.add_axes([.065, .30, .87, .49])
        axis.imshow(mpimg.imread(page['figure']['path'])); axis.axis('off')
        y = text(fig, page['conclusion'], .26, label='Conclusion:', shade=True)
        y = text(fig, page['figure'].get('caption', ''), y, size=8)
        text(fig, page.get('methods', ''), y, size=8)
        save(fig)

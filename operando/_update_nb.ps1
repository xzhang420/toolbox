$nbPath = 'c:\Software\toolbox\operando\operando_pipeline.ipynb'
$nb = Get-Content -Raw -Path $nbPath | ConvertFrom-Json
$newSource = @(
    '# === Capacity event images ===\n',
    '# Pixel size (physical size per pixel)\n',
    'pixel_size = 1.0  # TODO: set this\n',
    '\n',
    '# Capacity events (turning points + zeros)\n',
    'events = operando_data.find_capacity_turning_points_and_zeros(\n',
    '    echem_datetimes,\n',
    '    capacity_series=echem_thickness_series / conv_thickness,  # Capacity/mA.h\n',
    '    only_initial_zero=True,\n',
    ')\n',
    '\n',
    'if not events:\n',
    '    print("No capacity events found.")\n',
    'else:\n',
    '    event_times = [e["time"] for e in events]\n',
    '    img_indices = operando_data.find_nearest_image_indices(event_times, img_datetimes)\n',
    '\n',
    '    for e, idx in zip(events, img_indices):\n',
    '        print("{} {} -> image index {}".format(e["time"], e["labels"], idx))\n',
    '\n',
    '    # Plot images at these indices\n',
    '    import importlib.util\n',
    '    plots_path = os.path.join(os.path.dirname(notebook_dir), "plots", "plots.py")\n',
    '    plots_spec = importlib.util.spec_from_file_location("plots_mod", plots_path)\n',
    '    plots_mod = importlib.util.module_from_spec(plots_spec)\n',
    '    plots_spec.loader.exec_module(plots_mod)\n',
    '\n',
    '    plots_mod.plot_images(\n',
    '        img_series,\n',
    '        img_indices=img_indices,\n',
    '        pixel_size=pixel_size,\n',
    '        r0=roi["r0"],\n',
    '        x0=roi["x0"],\n',
    '        y0=roi["y0"],\n',
    '        cmap=cmap,\n',
    '    )\n'
)

$updated = $false
foreach ($cell in $nb.cells) {
    if ($cell.cell_type -eq 'code') {
        $src = $cell.source -join ''
        if ($src -match 'Capacity event images' -or $src -match 'find_capacity_turning_points_and_zeros') {
            $cell.source = $newSource
            $updated = $true
            break
        }
    }
}

if (-not $updated) {
    throw 'Capacity event cell not found.'
}

$nb | ConvertTo-Json -Depth 100 | Set-Content -Path $nbPath

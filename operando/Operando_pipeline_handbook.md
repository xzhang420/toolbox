# Operando Pipeline Guide

This guide is here:

- [Operando_pipeline_handbook.md](/data01/software/toolbox/operando/Operando_pipeline_handbook.md)

It is the quick user guide for:

- [operando_pipeline.ipynb](/data01/software/toolbox/operando/operando_pipeline.ipynb)
- [operando data.py](/data01/software/toolbox/operando/operando%20data.py)
- [operando plots.py](/data01/software/toolbox/operando/operando%20plots.py)

**What The Pipeline Does**

1. Corrects the raw neutron images using beam history and a reference image
2. Builds a background image and subtracts it
3. Loads electrochemistry and image timestamps
4. Finds capacity turning points and matches them to neutron images
5. Lets you preview, choose final images, save plots, and export a video

**Where Results Go**
Everything is saved automatically in:

- `results_<exp_name>/`

Main outputs:

- `obint_corrected_<exp_name>/`
- `uncorrected_<exp_name>/`
- `background_<exp_name>.tif`
- `final_<exp_name>/`
- `processing_metadata.json`
- `potential_<exp_name>.png`
- `nci_<exp_name>.png`
- `selected_image_summary_<exp_name>.txt`
- `<exp_name>.mp4`

**What You Usually Edit**
Before you start the notebook, first collect:

- beam history from Targetstrom on `https://sinqdata.psi.ch/`
- electrochemistry data exported from the potentiostat

Then set the main paths at the top of the notebook.

At the top of the notebook, set:

- `base_dir`
- `exp_dir`
- `beamhistory_path`
- `reference_dir`
- `reference_beamhistory_path`
- `echem_folder`
- `tpx_time`
- `n_tpx_per_image`
- `reference_thickness_um`

Most of the time, that is enough.

**Basic Workflow**

1. Run the imports cell
2. Set the experiment paths and parameters
3. Run obint correction
4. Run background-image creation
5. Run background subtraction
6. Load final images and EC data
7. Preview detected images
8. Preview the EC plot
9. Set:

```python
selected_image_indices = [...]
```

10. Save the final EC plot, image panel, and summary
11. Export the video

**Important Concepts**

Image timestamps
- Taken from TIFF metadata, not filenames

EC thickness
- Calculated as:
  `Capacity/mA.h * conv_thickness`

NCI thickness
- Mean pixel value inside `roi_small` for each final image

Turning points
- Found from the EC capacity/thickness curve
- Matched to the nearest neutron image

**Skip Behavior**
These processing stages are skipped if outputs already exist:

- obint correction
- background image
- background subtraction

When skipped, the notebook compares current settings to:

- `processing_metadata.json`

and prints warnings if parameters changed.

Plots and video are not skipped. They are overwritten when rerun.

**How To Pick Final Images**
Use the previews first, then edit:

```python
selected_image_indices = [12, 77]
```

These are image indices, so you can include:

- automatically detected images
- manually chosen images

**If You Move To A New Experiment**
Change only the top path variables:

- `exp_dir`
- `beamhistory_path`
- `reference_dir`
- `reference_beamhistory_path`
- `echem_folder`

Then rerun from the top.

**If Something Looks Wrong**

No images loaded
- Check `exp_dir`
- Check `final_<exp_name>/`

No EC data loaded
- Check `echem_folder`
- Check that the `.txt` files contain:
  - `time/s`
  - `Ewe/V`
  - `Capacity/mA.h`

Wrong match between turning point and image
- Check the warning printed in the event-context cell
- Check `tpx_time`
- Check `n_tpx_per_image`

No summary text file
- Rerun the final selected-output cell
- Make sure `selected_image_indices` is not empty

**Current Naming Convention**

- corrected images: `obint_corrected_<exp_name>`
- uncorrected copies: `uncorrected_<exp_name>`
- final images: `final_<exp_name>`
- EC plot: `potential_<exp_name>.png`
- NCI image panel: `nci_<exp_name>.png`
- summary text: `selected_image_summary_<exp_name>.txt`
- video: `<exp_name>.mp4`

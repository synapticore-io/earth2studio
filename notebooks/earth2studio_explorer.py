"""Earth2Studio Explorer — interactive weather & climate data notebook."""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    import tempfile

    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import requests
    import xarray as xr

    from earth2studio_client import Earth2StudioClient, InferenceRequest

    return Earth2StudioClient, InferenceRequest, go, json, mo, np, os, px, requests, tempfile, xr


@app.cell
def _(np):
    def slice_2d(da, lead_idx=None, ensemble_mode=None, subvar=None):
        if "time" in da.dims:
            da = da.isel(time=0)
        if lead_idx is not None and "lead_time" in da.dims:
            da = da.isel(lead_time=lead_idx)
        if "ensemble" in da.dims:
            if ensemble_mode == "std":
                da = da.std(dim="ensemble")
            elif ensemble_mode and ensemble_mode.isdigit():
                da = da.isel(ensemble=int(ensemble_mode))
            else:
                da = da.mean(dim="ensemble")
        if "sample" in da.dims:
            da = da.isel(sample=0)
        if "variable" in da.dims:
            if subvar and subvar in da.coords["variable"].values:
                da = da.sel(variable=subvar)
            else:
                da = da.isel(variable=0)
        return da.values.squeeze()

    def get_latlon(ds, fallback_shape):
        if "lat" in ds.data_vars:
            return ds["lat"].values, ds["lon"].values
        if "lat" in ds.coords and ds.coords["lat"].values.ndim == 1:
            return ds.coords["lat"].values, ds.coords["lon"].values
        return np.arange(fallback_shape[0]), np.arange(fallback_shape[1])

    def parse_params(raw, sch):
        out = {}
        for _k, _v in raw.items():
            _pt = sch.get("properties", {}).get(_k, {}).get("type", "string")
            try:
                if _pt == "integer":
                    out[_k] = int(_v)
                elif _pt == "number":
                    out[_k] = float(_v)
                elif _pt == "boolean":
                    out[_k] = _v.lower() in ("true", "1")
                elif _pt == "array" or _v.startswith("["):
                    import json as _json
                    out[_k] = _json.loads(_v.replace("'", '"'))
                elif _v in ("null", "None"):
                    out[_k] = None
                else:
                    out[_k] = _v
            except Exception:
                out[_k] = _v
        return out

    return get_latlon, parse_params, slice_2d


@app.cell
def _(np, px):
    from PIL import Image as _Image
    from matplotlib import cm as _cm
    from scipy.interpolate import interp1d as _interp1d
    import matplotlib as _mpl
    _mpl.use("Agg")

    def _lat_to_mercator(lat_deg):
        _r = np.radians(np.clip(lat_deg, -85, 85))
        return np.log(np.tan(np.pi / 4 + _r / 2))

    def make_map(vals, lat, lon, title, colorscale="turbo"):
        if lat.ndim == 1 and lon.ndim == 1:
            _lon = lon.copy().astype(np.float64)
            if np.any(_lon > 180):
                _lon = np.where(_lon > 180, _lon - 360, _lon)
                _sort = np.argsort(_lon)
                _lon = _lon[_sort]
                vals = vals[:, _sort] if vals.ndim == 2 else vals

            # Reproject to WebMercator y-spacing
            _merc_y = _lat_to_mercator(lat)
            _merc_uniform = np.linspace(_merc_y[0], _merc_y[-1], len(lat))
            _vals_merc = _interp1d(_merc_y, vals, axis=0, kind="nearest", fill_value="extrapolate")(_merc_uniform)

            # Render as RGBA image
            _vmin = float(np.nanpercentile(vals, 2))
            _vmax = float(np.nanpercentile(vals, 98))
            _norm = np.clip((_vals_merc - _vmin) / max(_vmax - _vmin, 1e-10), 0, 1)
            _rgba = (_cm.turbo(_norm) * 255).astype(np.uint8)
            _rgba[:, :, 3] = 200
            _img = _Image.fromarray(_rgba, "RGBA")

            # Coordinates: same corners as datashader example
            _coordinates = [
                [float(_lon[0]), float(lat[0])],
                [float(_lon[-1]), float(lat[0])],
                [float(_lon[-1]), float(lat[-1])],
                [float(_lon[0]), float(lat[-1])],
            ]

            _center_lat = float((lat[0] + lat[-1]) / 2)
            _center_lon = float((_lon[0] + _lon[-1]) / 2)
            _max_range = max(abs(float(lat[0]) - float(lat[-1])), abs(float(_lon[-1]) - float(_lon[0])))
            _zoom = 1 if _max_range > 180 else 2 if _max_range > 90 else 3 if _max_range > 45 else 4 if _max_range > 20 else 5
            _clean = title.split("<br>")[0].replace("<b>", "").replace("</b>", "")

            # Invisible scatter for hover
            _sy = max(1, len(lat) // 50)
            _sx = max(1, len(_lon) // 50)
            _la = np.repeat(lat[::_sy], len(_lon[::_sx]))
            _lo = np.tile(_lon[::_sx], len(lat[::_sy]))
            _hv = vals[::_sy, ::_sx].ravel()

            _fig = px.scatter_map(lat=_la, lon=_lo, color=_hv,
                color_continuous_scale="turbo", range_color=[_vmin, _vmax],
                zoom=_zoom, center=dict(lat=_center_lat, lon=_center_lon),
                labels={"color": _clean})
            _fig.update_traces(marker=dict(size=12, opacity=0),
                hovertemplate="Lat: %{lat:.1f}<br>Lon: %{lon:.1f}<br>Value: %{marker.color:.3g}<extra></extra>")
            _fig.update_layout(
                map_style="carto-positron",
                map_layers=[{"sourcetype": "image", "source": _img,
                             "coordinates": _coordinates, "opacity": 0.8}],
                title=title, height=550, margin=dict(l=0, r=0, t=50, b=0),
            )
        else:
            _fig = px.imshow(vals, color_continuous_scale=colorscale, aspect="auto")
            _fig.update_layout(title=title, height=500)
        return _fig

    return (make_map,)


@app.cell
def _(mo):
    mo.md("""
    # Earth2Studio Explorer
    """)
    return


@app.cell
def _(mo):
    api_url = mo.ui.text(value="http://localhost:8000", label="API URL")
    return (api_url,)


@app.cell
def _(api_url, mo, requests):
    try:
        _r = requests.get(f"{api_url.value}/v1/workflows", timeout=5)
        _r.raise_for_status()
        _wfs = _r.json()["workflows"]
        workflow_dropdown = mo.ui.dropdown(
            options={f"{_k} \u2014 {_v}": _k for _k, _v in _wfs.items()},
            label="Workflow",
        )
        _out = mo.vstack([api_url, mo.callout(mo.md(f"**Connected** \u2014 {len(_wfs)} workflows"), kind="success"), workflow_dropdown])
    except Exception as _e:
        workflow_dropdown = mo.ui.dropdown(options={}, label="Workflow")
        _out = mo.vstack([api_url, mo.callout(mo.md(f"**Disconnected**: `{_e}`"), kind="danger")])
    _out
    return (workflow_dropdown,)


@app.cell
def _(api_url, json, mo, requests, workflow_dropdown):
    mo.stop(not workflow_dropdown.value)
    _name = workflow_dropdown.value
    try:
        _r = requests.get(f"{api_url.value}/v1/workflows/{_name}/schema", timeout=5)
        _r.raise_for_status()
        schema = _r.json()
    except Exception:
        schema = {"properties": {}}
    _inputs = {}
    for _k, _p in schema.get("properties", {}).items():
        _d = _p.get("default", "")
        if isinstance(_d, list):
            _d = json.dumps(_d)
        _inputs[_k] = mo.ui.text(value=str(_d), label=_k)
    param_form = mo.ui.dictionary(_inputs)
    submit_btn = mo.ui.run_button(label="Run Forecast")
    mo.vstack([mo.md(f"### `{_name}`"), param_form, submit_btn])
    return param_form, schema, submit_btn


@app.cell
def _(
    Earth2StudioClient,
    InferenceRequest,
    api_url,
    mo,
    os,
    param_form,
    parse_params,
    schema,
    submit_btn,
    tempfile,
    workflow_dropdown,
    xr,
):
    mo.stop(not submit_btn.value)

    _name = workflow_dropdown.value
    _params = parse_params(param_form.value, schema)
    _client = Earth2StudioClient(api_url.value, workflow_name=_name)
    _req = InferenceRequest(parameters=_params)
    _req_result = _client.run_inference_sync(_req)

    # Download zarr to local temp dir (avoids zarr v3 remote-read issues)
    _tmpdir = tempfile.mkdtemp()
    for _f in _req_result.output_files:
        _rel = _f.path.split("/", 1)[1] if "/" in _f.path else _f.path
        _local = os.path.join(_tmpdir, _rel)
        os.makedirs(os.path.dirname(_local), exist_ok=True)
        _content = _client.download_result(_req_result, _f.path)
        with open(_local, "wb") as _fh:
            _fh.write(_content.getvalue())
    ds = xr.open_zarr(os.path.join(_tmpdir, "results.zarr"), consolidated=True)
    mo.callout(mo.md(f"**Done** `{_req_result.request_id}` — {dict(ds.sizes)}"), kind="success")
    return (ds,)


@app.cell
def _(ds, mo):
    _dvars = [_v for _v in ds.data_vars if _v not in ("lat", "lon")]
    var_select = mo.ui.dropdown(options=_dvars, value=_dvars[0] if _dvars else None, label="Variable")
    subvar_select = None
    _fv = _dvars[0] if _dvars else None
    if _fv and "variable" in ds[_fv].dims:
        _sv = [str(_v) for _v in ds.coords["variable"].values]
        subvar_select = mo.ui.dropdown(options=_sv, value=_sv[0], label="Sub-Variable")
    lead_slider = None
    if "lead_time" in ds.dims and ds.dims["lead_time"] > 1:
        lead_slider = mo.ui.slider(start=0, stop=ds.dims["lead_time"] - 1, value=0, label="Lead Time")
    ensemble_select = None
    if "ensemble" in ds.dims and ds.dims["ensemble"] > 1:
        _opts = {"Mean": "mean", "Std Dev": "std"}
        _opts.update({f"Member {_j}": str(_j) for _j in range(ds.dims["ensemble"])})
        ensemble_select = mo.ui.dropdown(options=_opts, value="mean", label="Ensemble")
    _c = [var_select]
    if subvar_select is not None:
        _c.append(subvar_select)
    if lead_slider is not None:
        _c.append(lead_slider)
    if ensemble_select is not None:
        _c.append(ensemble_select)
    mo.hstack(_c, gap=1)
    return ensemble_select, lead_slider, subvar_select, var_select


@app.cell
def _(
    ds,
    ensemble_select,
    get_latlon,
    lead_slider,
    make_map,
    mo,
    np,
    slice_2d,
    subvar_select,
    var_select,
    workflow_dropdown,
):
    mo.stop(not var_select.value)
    _var = var_select.value
    _sv = subvar_select.value if subvar_select is not None else None
    _label = _sv or _var
    _vals = slice_2d(
        ds[_var],
        lead_idx=lead_slider.value if lead_slider is not None else None,
        ensemble_mode=ensemble_select.value if ensemble_select is not None else None,
        subvar=_sv,
    )
    _lat, _lon = get_latlon(ds, _vals.shape)

    # Build descriptive title: Variable | Workflow | Date | Lead Time | Ensemble
    _wf_name = workflow_dropdown.value or ""
    _time_str = ""
    if "time" in ds.coords:
        _t = ds.coords["time"].values[0]
        _time_str = str(_t)[:16].replace("T", " ")
    _title = f"<b>{_label}</b>"
    if _wf_name:
        _title += f" — {_wf_name}"
    if _time_str:
        _title += f"<br><sup>Init: {_time_str} UTC"
    if lead_slider is not None and "lead_time" in ds.coords:
        _lt = ds.coords["lead_time"].values[lead_slider.value]
        _lt_h = int(_lt / np.timedelta64(1, "h")) if isinstance(_lt, np.timedelta64) else lead_slider.value
        _title += f" | +{_lt_h}h"
    if ensemble_select is not None:
        _title += f" | Ens: {ensemble_select.value}"
    if "time" in ds.coords:
        _title += "</sup>"

    _fig = make_map(_vals, _lat, _lon, _title)
    _fig
    return


@app.cell
def _(mo):
    enable_compare = mo.ui.switch(label="Compare Mode", value=False)
    enable_compare
    return (enable_compare,)


@app.cell
def _(api_url, enable_compare, json, mo, requests, workflow_dropdown):
    mo.stop(not enable_compare.value)
    _name = workflow_dropdown.value
    mo.stop(not _name)
    try:
        _r = requests.get(f"{api_url.value}/v1/workflows/{_name}/schema", timeout=5)
        _r.raise_for_status()
        compare_schema = _r.json()
    except Exception:
        compare_schema = {"properties": {}}
    _inputs = {}
    for _k, _p in compare_schema.get("properties", {}).items():
        _d = _p.get("default", "")
        if isinstance(_d, list):
            _d = json.dumps(_d)
        _inputs[_k] = mo.ui.text(value=str(_d), label=_k)
    compare_form = mo.ui.dictionary(_inputs)
    compare_submit = mo.ui.run_button(label="Run Compare")
    mo.vstack([mo.md(f"### Compare: `{_name}`"), compare_form, compare_submit])
    return compare_form, compare_schema, compare_submit


@app.cell
def _(
    Earth2StudioClient,
    InferenceRequest,
    api_url,
    compare_form,
    compare_schema,
    compare_submit,
    enable_compare,
    mo,
    os,
    parse_params,
    tempfile,
    workflow_dropdown,
    xr,
):
    mo.stop(not enable_compare.value)
    mo.stop(not compare_submit.value)
    _name = workflow_dropdown.value
    _params = parse_params(compare_form.value, compare_schema)
    _client = Earth2StudioClient(api_url.value, workflow_name=_name)
    _req = InferenceRequest(parameters=_params)
    _req_result = _client.run_inference_sync(_req)
    _tmpdir = tempfile.mkdtemp()
    for _f in _req_result.output_files:
        _rel = _f.path.split("/", 1)[1] if "/" in _f.path else _f.path
        _local = os.path.join(_tmpdir, _rel)
        os.makedirs(os.path.dirname(_local), exist_ok=True)
        _content = _client.download_result(_req_result, _f.path)
        with open(_local, "wb") as _fh:
            _fh.write(_content.getvalue())
    ds_compare = xr.open_zarr(os.path.join(_tmpdir, "results.zarr"), consolidated=True)
    mo.callout(mo.md(f"**Compare done** `{_req_result.request_id}`"), kind="success")
    return (ds_compare,)


@app.cell
def _(
    ds,
    ds_compare,
    enable_compare,
    get_latlon,
    make_map,
    mo,
    slice_2d,
    subvar_select,
    var_select,
):
    mo.stop(not enable_compare.value)
    mo.stop(not var_select.value)
    _var = var_select.value
    mo.stop(_var not in ds.data_vars or _var not in ds_compare.data_vars)
    _sv = subvar_select.value if subvar_select is not None else None
    _v1 = slice_2d(ds[_var], subvar=_sv)
    _v2 = slice_2d(ds_compare[_var], subvar=_sv)
    _lat, _lon = get_latlon(ds, _v1.shape)
    _label = _sv or _var
    mo.hstack([
        make_map(_v1, _lat, _lon, f"{_label} \u2014 Primary"),
        make_map(_v2, _lat, _lon, f"{_label} \u2014 Compare"),
    ], gap=0.5)
    return


if __name__ == "__main__":
    app.run()
